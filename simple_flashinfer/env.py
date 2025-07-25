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
        'microsleep', type=float,
        default=float(os.environ.get('DYNAMIC_BATCHING_MICROSLEEP', '1e-4')),
        help='microsleep to group dynamic batching, 1 / 1e-4 = 10k steps for second (default: %(default)s, env: DYNAMIC_BATCHING_MICROSLEEP)'
    )
    parser.add_argument(
        '--model',
        default=os.environ.get('MODEL', 'Qwen/Qwen2.5-0.5B-Instruct'),
        help='Model type (default: %(default)s, env: MODEL)'
    )

    args = parser.parse_args()

    args.device = 'cuda'
    return args


args = parse_arguments()

logging.basicConfig(level=args.loglevel)

logging.info(f'Serving app using {args}')