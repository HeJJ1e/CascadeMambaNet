import argparse

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', choices=('bf', 'zaha'), required=True)
    (arguments, remaining) = parser.parse_known_args()
    if arguments.dataset == 'bf':
        from engine.bf_pipeline import main as run_pipeline
    else:
        from engine.zaha_pipeline import main as run_pipeline
    run_pipeline(remaining)
if __name__ == '__main__':
    main()
