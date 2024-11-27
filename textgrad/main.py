from dataclasses import dataclass, field
from typing import Dict, List, Union, Any, Callable


@dataclass
class Variable:
    string: str
    key: str
    requires_grad: bool = False
    grad: str = None
    predecessors: List[Any] = field(default_factory=list)
    conversation: str = ''

    def __str__(self):
        return f'[{self.key}] {self.string}'

    def backward(self, grad: str = None):
        print('Backwarding', self.key)
        self.grad = grad
        global LLM
        for p in self.predecessors or []:
            if p.requires_grad:
                if grad is None:
                    template = f'''
Here is a conversation with an LLM:

{self.conversation}

Based on the evaluations, explain how to improve [{p.key}].
'''
                else:
                    template = f'''
Here is a conversation with an LLM:

{p.conversation}

Below are the criticisms on [{self.key}]:

{self.grad}

Explain how to improve [{p.key}].
'''

                grad = LLM.get_response(template)
                p.backward(grad)

    def zero_grad(self):
        self.grad = None


class BaseLLM:
    def __init__(self):
        pass

    def get_response(self, msg: str) -> str:
        raise NotImplementedError

    def __call__(self, msg: str, variables: Dict[Any, Variable] = None, key: str = None) -> Variable:
        requires_grad = any([v.requires_grad for v in variables.values()])
        full_msg = msg.format(**variables)
        resp = self.get_response(full_msg)
        conversation = f'{full_msg}\n\nLLM: {resp}'
        predecessors = list(variables.values())
        var = Variable(resp,
                       requires_grad=requires_grad,
                       key=key,
                       predecessors=predecessors,
                       conversation=conversation)
        return var


LLM: BaseLLM = NotImplemented


def setLLM(llm: BaseLLM):
    global LLM
    LLM = llm


@dataclass
class Optimizer:
    parameters: List[Variable]
    template: str = '''
There is a variable: {x}

Below are the criticisms on it:
{grad}

Incorporate the criticisms, and produce a new variable.
'''

    def step(self):
        global LLM
        for p in self.parameters:
            assert p.grad
            msg = self.template.format(x=p, grad=p.grad)
            new_var = LLM.get_response(msg)
            p.string = new_var
