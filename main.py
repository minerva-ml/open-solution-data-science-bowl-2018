import click
from src.pipeline_manager import PipelineManager

pipeline_manager = PipelineManager()

@click.group()
def action():
    pass


@action.command()
def prepare_metadata():
    pipeline_manager.prepare_metadata()


@action.command()
def prepare_masks():
    pipeline_manager.prepare_masks()


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.2, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train(pipeline_name, validation_size, dev_mode):
    pipeline_manager.train(pipeline_name, validation_size, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.2, type=str, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def evaluate(pipeline_name, validation_size, dev_mode):
    pipeline_manager.evaluate(pipeline_name, validation_size, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def predict(pipeline_name, dev_mode):
    pipeline_manager.predict(pipeline_name, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train_evaluate_predict(pipeline_name, validation_size, dev_mode):
    pipeline_manager.train(pipeline_name, validation_size, dev_mode)
    pipeline_manager.evaluate(pipeline_name, validation_size, dev_mode)
    pipeline_manager.predict(pipeline_name, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def train_evaluate(pipeline_name, validation_size, dev_mode):
    pipeline_manager.train(pipeline_name, validation_size, dev_mode)
    pipeline_manager.evaluate(pipeline_name, validation_size, dev_mode)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
@click.option('-d', '--dev_mode', help='if true only a small sample of data will be used', is_flag=True, required=False)
def evaluate_predict(pipeline_name, validation_size, dev_mode):
    pipeline_manager.evaluate(pipeline_name, validation_size, dev_mode)
    pipeline_manager.predict(pipeline_name, dev_mode)


if __name__ == "__main__":
    action()
