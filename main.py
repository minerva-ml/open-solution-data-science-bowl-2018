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
def train(pipeline_name, validation_size):
    pipeline_manager.train(pipeline_name, validation_size)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default='0.2', required=False)
def evaluate(pipeline_name, validation_size):
    pipeline_manager.evaluate(pipeline_name, validation_size)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
def predict(pipeline_name):
    pipeline_manager.predict(pipeline_name)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
def train_evaluate_predict_pipeline(pipeline_name, validation_size):
    pipeline_manager.train(pipeline_name, validation_size)
    pipeline_manager.evaluate(pipeline_name, validation_size)
    pipeline_manager.predict(pipeline_name)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
def train_evaluate_pipeline(pipeline_name, validation_size):
    pipeline_manager.train(pipeline_name, validation_size)
    pipeline_manager.evaluate(pipeline_name, validation_size)


@action.command()
@click.option('-p', '--pipeline_name', help='pipeline to be trained', required=True)
@click.option('-v', '--validation_size', help='percentage of training used for validation', default=0.1, required=False)
def evaluate_predict_pipeline(pipeline_name, validation_size):
    pipeline_manager.evaluate(pipeline_name, validation_size)
    pipeline_manager.predict(pipeline_name)


if __name__ == "__main__":
    action()
