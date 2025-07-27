import argparse
import logging
import os
import torch


def parse_arguments():
    parser = argparse.ArgumentParser(description='Configuration parser')

    parser.add_argument(
        '--host', type=str, default=os.environ.get('HOSTNAME', '0.0.0.0'),
        help='host name to host the app (default: %(default)s, env: HOSTNAME)'
    )
    parser.add_argument(
        '--port', type=int, default=int(os.environ.get('PORT', '7088')),
        help='port to host the app (default: %(default)s, env: PORT)'
    )
    parser.add_argument(
        '--loglevel', default=os.environ.get('LOGLEVEL', 'INFO').upper(),
        help='Logging level (default: %(default)s, env: LOGLEVEL)'
    )
    parser.add_argument(
        '--reload', type=lambda x: x.lower() == 'true',
        default=os.environ.get('reload', 'false').lower() == 'true',
        help='Enable hot loading (default: %(default)s, env: RELOAD)'
    )
    parser.add_argument(
        '--microsleep', type=float,
        default=float(os.environ.get('MICROSLEEP', '1e-4')),
        help='microsleep to group batching to reduce CPU burden, 1 / 1e-4 = 10k steps for second (default: %(default)s, env: MICROSLEEP)'
    )
    parser.add_argument(
        '--max_sequence', type=float,
        default=float(os.environ.get('MAX_SEQUENCE', '128')),
        help='max sequence aka batch size per filling or decoding (default: %(default)s, env: MAX_SEQUENCE)'
    )
    parser.add_argument(
        '--memory_utilization', type=float,
        default=float(os.environ.get('MEMORY_UTILIZATION', '0.9')),
        help='memory utilization on free memory after load the model for automatic number of paging for paged attention (default: %(default)s, env: MEMORY_UTILIZATION)'
    )
    parser.add_argument(
        '--compare-sdpa-prefill', type=lambda x: x.lower() == 'true',
        default=os.environ.get('COMPARE_SDPA_PREFILL', 'false').lower() == 'true',
        help='Compare FlashInfer attention output with SDPA during prefill (default: %(default)s, env: COMPARE_SDPA_PREFILL)'
    )
    parser.add_argument(
        '--model',
        default=os.environ.get('MODEL', 'meta-llama/Llama-3.2-1B-Instruct'),
        help='Model type (default: %(default)s, env: MODEL)'
    )
    parser.add_argument(
        '--torch_dtype',
        default=os.environ.get('TORCH_DTYPE', 'float16'),
        help='Model type (default: %(default)s, env: TORCH_DTYPE)'
    )

    args = parser.parse_args()

    if args.torch_dtype not in {'float16', 'bfloat16'}:
        raise ValueError('`--torch_dtype` only support `float16` or `bfloat16`')

    args.device = 'cuda'
    args.torch_dtype = getattr(torch, args.torch_dtype)
    return args


args = parse_arguments()

logging.basicConfig(level=args.loglevel)

logging.info(f'Serving app using {args}')