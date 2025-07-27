from locust import HttpUser, task
from locust import events

"""
Make sure already running this,

python3 -m simple_flashinfer.main --host 0.0.0.0 --port 7088 --model {model}
"""

class HelloWorldUser(HttpUser):

    host = "http://127.0.0.1:7088"

    @task
    def hello_world(self):

        json_data = {
            'model': 'model',
            'temperature': 0.0,
            'top_p': 0.95,
            'top_k': 50,
            'max_tokens': 256,
            'truncate': 2048,
            'repetition_penalty': 1,
            'stop': [],
            'messages': [
                {
                    'role': 'user',
                    'content': 'How to code python3',
                },
            ],
            'stream': False,
        }
        r = self.client.post('/chat/completions', json=json_data)