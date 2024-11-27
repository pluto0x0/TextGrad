import textgrad as tg

code_str = '''
def maxSubArray(nums: list):
    max_sum = 0
    n = len(nums)
    for i in range(n):
        for j in range(i, n):
            max_sum = max(max_sum, sum(nums[i:j+1]))
    return max_sum
'''

question = tg.Variable(
    'Given a array of integers, return the maximum sum of a subarray of it.', key='question')
code = tg.Variable(code_str, key='code', requires_grad=True)

LLM = tg.models.OpenAI()
tg.setLLM(LLM)

evaluation_str = '''
You are given the following question and solution:

{question}

{code}

Please analyze this solution: a) Is the solution correct? b) Is the solution efficient?
'''

loss = LLM(evaluation_str, variables={
           'question': question, 'code': code}, key='evaluation')

loss.backward()

print(loss.predecessors)

print(code.grad)

opt = tg.Optimizer([code])

opt.step()

print(code)
