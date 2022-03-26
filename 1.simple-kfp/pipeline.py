"""
https://github.com/kubeflow/examples/blob/master/pipelines/simple-notebook-pipeline/Simple%20Notebook%20Pipeline.ipynb
"""
import kfp.dsl as dsl
from kfp import components

BASE_IMAGE = "python:3.8.2"
EXPERIMENT_NAME = "add"


# Create component
@dsl.python_component(
    name='add_op',
    description='adds two numbers',
    base_image=BASE_IMAGE
)
def add(a: float, b: float) -> float:
    print("{} + {} = {}".format(a, b, a+b))
    return a + b


add_op = components.func_to_container_op(
    add,
    base_image=BASE_IMAGE
)


# Build a pipeline using component
@dsl.pipeline(
    name='Calculation pipeline',
    description='simple pipeline'
)
def cal_pipeline(a: float,
                 b: float):
    add_task = add_op(a, 4)
    add_task2 = add_op(a, b)
    add_task3 = add_op(add_task.output, add_task2.output)


if __name__ == "__main__":
    from kfp import compiler
    # Compile and run the pipeline
    arguments = {'a': '7', 'b': '8'}
    pipeline_filename = cal_pipeline.__name__ + '.pipeline.zip'
    compiler.Compiler().compile(cal_pipeline, pipeline_filename)
