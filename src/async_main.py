#!/usr/bin/env python3
import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

from lime.common.controllers.parse import parse_to_obj
from lime.common.models.internal import SheetSchema, SheetOutputSchema
from lime.common.models.utils import get_lime_version
from lime.common.views.msg.eval import SheetProgressMsg
from lime.common.inference.interface import get_infer_obj, ModelObjVariant
from lime.common.models.state import ExecSettings
from lime.common.models.errs import BaseQuietError

@dataclass
class PerformanceMetrics:
    total_eval_time: float
    avg_eval_time: float
    total_tokens: int
    cache_hits: int
    cache_misses: int
    successful_evals: int
    failed_evals: int
    
    def __str__(self):
        total_evals = self.successful_evals + self.failed_evals
        success_rate = (self.successful_evals / total_evals * 100) if total_evals > 0 else 0
        cache_hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return (f"Performance Metrics:\n"
                f"  Total Eval Time: {self.total_eval_time:.2f}s\n"
                f"  Avg Eval Time: {self.avg_eval_time:.2f}s\n"
                f"  Total Tokens: {self.total_tokens}\n"
                f"  Cache Hits: {self.cache_hits} ({cache_hit_rate:.1f}%)\n"
                f"  Cache Misses: {self.cache_misses}\n"
                f"  Success Rate: {self.successful_evals}/{total_evals} ({success_rate:.1f}%)\n")

class AsyncEvalSheet:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.metrics = PerformanceMetrics(0, 0, 0, 0, 0, 0, 0)
    
    def _count_tokens(self, text: str, infer_obj: ModelObjVariant) -> int:
        try:
            return infer_obj.count_tokens(text)
        except Exception:
            return len(text.split())
    
    async def _prompt_model_async(self, infer_obj: ModelObjVariant, prompt_sys: str, prompt_usr: str, **gen_params) -> tuple:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._prompt_model_sync,
            infer_obj,
            prompt_sys,
            prompt_usr,
            **gen_params
        )
    
    def _prompt_model_sync(self, infer_obj: ModelObjVariant, prompt_sys: str, prompt_usr: str, **gen_params) -> tuple:
        try:
            return infer_obj.prompt_model(prompt_sys=prompt_sys, prompt_usr=prompt_usr, **gen_params)
        except Exception as e:
            return None, str(e)
    
    async def _eval_question_async(self, question, infer_obj: ModelObjVariant, sheet_gen_params: dict, progress, output: SheetOutputSchema, question_idx: int) -> Optional[dict]:
        t0 = time.time()
        
        # Cache system prompt
        sys_prompt = question.text_sys
        cache_key = f"sys_{hash(sys_prompt) % 10000}"
        
        if ExecSettings.prompt_cache and sys_prompt:
            cached = ExecSettings.prompt_cache.get(cache_key)
            if cached:
                ExecSettings.prompt_cache.cache_hits += 1
                sys_prompt = cached
            else:
                ExecSettings.prompt_cache.cache_misses += 1
                ExecSettings.prompt_cache.set(cache_key, sys_prompt)
        
        # Async LLM call
        completion, error = await self._prompt_model_async(infer_obj, sys_prompt, question.text_usr, **sheet_gen_params)
        eval_time = time.time() - t0
        
        # Token counting
        ntokens_usr = self._count_tokens(question.text_usr, infer_obj)
        ntokens_sys = self._count_tokens(sys_prompt, infer_obj)
        ntokens_cmp = self._count_tokens(completion or "", infer_obj) if not error else 0
        
        # Build output
        question_output = {
            "name": question.name,
            "meta_data": question.meta,
            "gen_params": sheet_gen_params,
            "ground_truth": question.answer,
            "question_sys": question.text_sys,
            "question_usr": question.text_usr,
            "completion": completion,
            "error": str(error) if error else None,
            "eval_time": eval_time,
            "ntokens": {
                "usr": ntokens_usr,
                "sys": ntokens_sys,
                "cmp": ntokens_cmp
            }
        }
        
        # Update metrics
        self.metrics.total_eval_time += eval_time
        self.metrics.total_tokens += (ntokens_usr + ntokens_sys + ntokens_cmp)
        
        if error:
            self.metrics.failed_evals += 1
        else:
            self.metrics.successful_evals += 1
            grading_output = grade_answer(completion, question.answer)
            question_output["grading"] = grading_output
        
        # Update progress
        progress.post_prompt(question_output)
        
        # Add to output
        output["questions"].append(question_output)
        
        return question_output
    
    async def eval_sheet_async(self, sheet_obj: SheetSchema, infer_obj: ModelObjVariant, run_id: str, verbose_level: int = 0) -> SheetOutputSchema:
        progress = SheetProgressMsg(verbose_level=verbose_level)
        
        sheet_gen_params = extract_gen_params(sheet_obj.run_id)
        infer_obj.update_gen_params(sheet_gen_params)
        
        output = {
            "header": {
                "sheet_name": sheet_obj.name,
                "sheet_fn": sheet_obj.sheet_fn,
                "name_model": infer_obj.model_name,
                "infer_params": infer_obj.get_gen_params(),
                "lime_version": get_lime_version(),
                "start_time": time.time(),
            },
            "questions": [],
            "performance": {}
        }
        
        progress.pre_loop(sheet_obj)
        
        # Process questions in parallel
        tasks = []
        for idx, question in enumerate(sheet_obj.questions):
            task = self._eval_question_async(question, infer_obj, sheet_gen_params, progress, output, idx)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate final metrics
        total_questions = len(sheet_obj.questions)
        if total_questions > 0:
            self.metrics.avg_eval_time = self.metrics.total_eval_time / total_questions
        
        output["performance"] = {
            "total_eval_time": self.metrics.total_eval_time,
            "avg_eval_time": self.metrics.avg_eval_time,
            "total_tokens": self.metrics.total_tokens,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "successful_evals": self.metrics.successful_evals,
            "failed_evals": self.metrics.failed_evals,
            "success_rate": self.metrics.successful_evals / total_questions if total_questions > 0 else 0
        }
        
        progress.post_loop(output)
        
        return output

async def batch_eval_async(
    sheet_fns: List[str],
    model_name: str,
    run_id: str,
    dry_run: bool = False,
    use_prompt_cache: bool = True,
    verbose_level: int = 0,
) -> None:
    
    progress = SheetProgressMsg(verbose_level=verbose_level)
    progress.pre_loop(sheet_fns=sheet_fns)
    
    # Initialize async evaluator
    async_eval = AsyncEvalSheet()
    
    try:
        infer_constructor_args = {
            'use_prompt_cache': use_prompt_cache,    
        }
        infer_obj = get_infer_obj(model_name, **infer_constructor_args)
        progress.infer_init(infer_obj, infer_obj.check_valid())
    
    except Exception as e:
        raise BaseQuietError(f'Error creating infer_obj: {str(e)}')
    
    # Process sheets in parallel
    tasks = []
    for sheet_fn in sheet_fns:
        task = _process_sheet_async(sheet_fn, infer_obj, run_id, dry_run, verbose_level, async_eval)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error processing sheet {sheet_fns[i]}: {result}")
        else:
            output_fp = make_output_fp(sheet_fns[i], model_name, run_id)
            with open(output_fp, 'w', encoding='utf-8', errors='replace') as f:
                f.write(json.dumps(result, indent=2))
    
    progress.post_loop(None)

async def _process_sheet_async(sheet_fn: str, infer_obj: ModelObjVariant, run_id: str, dry_run: bool, verbose_level: int, async_eval: AsyncEvalSheet):
    sheet_obj = parse_to_obj(sheet_fn)
    
    progress = SheetProgressMsg(verbose_level=verbose_level)
    progress.pre_sheet(sheet_obj)
    
    output = await async_eval.eval_sheet_async(sheet_obj, infer_obj, run_id, verbose_level)
    
    return output

def make_output_fp(sheet_fn: str, model_name: str, run_id: str) -> str:
    sheet_dir = os.path.dirname(sheet_fn)
    fn = os.path.basename(sheet_fn)
    input_prefix = ExecSettings.input_sheet_prefix
    output_prefix = ExecSettings.output_sheet_prefix
    fn = fn.replace('.md', '')
    fn = fn.replace(input_prefix, '')
    sep = '-' if (fn[0].isalpha() or fn[0].isdigit()) else ''
    output_fn = f'{output_prefix}{sep}{fn}-{model_name}-{run_id}.json'
    output_fp = os.path.join(sheet_dir, output_fn)
    return str(output_fp)

def make_tmp_output_fp(output_fp: str) -> Optional[str]:
    if ExecSettings.save_tmp_file:
        return str(os.path.join(
            os.path.dirname(output_fp),
            f'tmp-{os.path.basename(output_fp)}'
        ))
    return None

def cleanup_tmp(tmp_output_fp: Optional[str]) -> None:
    if tmp_output_fp is None:
        return
    if os.path.exists(tmp_output_fp):
        try:
            os.remove(tmp_output_fp)
        except Exception as e:
            err_msg = f'Error removing tmp: {tmp_output_fp}: {e}'
            if BaseQuietError.debug_mode:
                BaseQuietError(err_msg)
            else:
                print(err_msg)

def continue_or_exit() -> None:
    try:
        print('\n')
        val = input('Press Enter to continue, any other key to quit...')
        if val != '':
            sys.exit(1)
        else:
            return
    except KeyboardInterrupt:
        print('Keyboard Interrupt.')
        sys.exit(1)
    except Exception as e:
        raise BaseQuietError(f'Error trying to continue: {str(e)}')

def filter_input_sheet(fn: str, fn_keyword: str = ExecSettings.input_sheet_prefix, fn_ext: str = '.md') -> bool:
    return (fn_keyword in fn) and (fn_ext in fn)

def filter_input_sheets(fns: List[str]) -> List[str]:
    return [fn for fn in fns if filter_input_sheet(fn)]

def get_sheet_fns(input_paths: List[str]) -> List[str]:
    all_sheet_fns = []
    for input_path in input_paths:
        if os.path.isfile(input_path):
            matched_files = [input_path]
        elif input_path == '.':
            matched_files = filter_input_sheets(glob.glob('*'))
        elif os.path.isdir(input_path):
            input_path = os.path.join(input_path, '*')
            matched_files = filter_input_sheets(glob.glob(input_path))
        else:
            matched_files = filter_input_sheets(glob.glob(input_path))
        if matched_files:
            all_sheet_fns += matched_files
    if len(all_sheet_fns) == 0:
        raise BaseQuietError(f'No input files found in: {input_paths}')
    return all_sheet_fns

def setup_parser(parser):
    parser.add_argument('input_paths', metavar='N', type=str, nargs='+', help='an input path or glob pattern')
    parser.add_argument('-m', '--model_name', type=str)
    parser.add_argument('-n', '--model_nick_name', type=str)
    parser.add_argument('-y', '--dry_run', action='store_true')
    parser.add_argument('-v', '--verbose', action='count')
    parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('-b', '--debug', action='store_true')

def main(args):
    args = vars(args)

    if args.get('debug'):
        BaseQuietError.debug_mode = True

    if args.get('verbose'):
        BaseQuietError.debug_mode = True

    sheet_fns = get_sheet_fns(args['input_paths'])

    model_name = args.get('model_name') or ExecSettings.model_nick_name
    verbose_level = args.get('verbose') or ExecSettings.verbose
    dry_run = args.get('dry_run') or False

    run_id = uuid.uuid4().hex[:ExecSettings.uuid_digits]

    use_prompt_cache = ExecSettings.use_prompt_cache

    # Initialize prompt cache if enabled
    if use_prompt_cache:
        ExecSettings.prompt_cache = {
            'cache': {},
            'hits': 0,
            'misses': 0
        }
    
    # Run async batch evaluation
    asyncio.run(batch_eval_async(
        sheet_fns=sheet_fns,
        model_name=model_name,
        run_id=run_id,
        dry_run=dry_run,
        use_prompt_cache=use_prompt_cache,
        verbose_level=verbose_level,
    ))

if __name__ == "__main__":
    import argparse
    import glob
    import uuid
    import json
    
    parser = argparse.ArgumentParser()
    setup_parser(parser)
    
    main(parser.parse_args())