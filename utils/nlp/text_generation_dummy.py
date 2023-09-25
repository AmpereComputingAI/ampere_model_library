import random

from utils.helpers import DatasetStub


class TextGenerationDummyDataset(DatasetStub):

    def __init__(self, batch_size: int, tokenize_func):
        self._idx = 0
        self.available_instances = 100000

        self.__batch_size = batch_size
        self.__tokenize_func = tokenize_func
        self.__current_inputs = None

    def get_input(self):
        random.seed(42)
        prompts = ["hi, how are you?",
                   "Generate a script for a 30-second commercial promoting our new product",
                   "Write a persuasive email to convince potential customers to try our service",
                   "Create a list of frequently asked questions for our customer service team",
                   "Generate a summary of our company’s mission and values",
                   "Write a script for a training video on how to use our software",
                   "Create a list of potential blog post ideas for our company’s website",
                   "Generate a press release announcing our company’s latest partnership",
                   "Write a script for a video testimonial from a satisfied customer",
                   "Create a list of keywords to optimize our website for search engines",
                   "Generate a script for a social media video showcasing our company culture",
                   "Write a script for a explainer video about our new product",
                   "Create a list of potential influencers to collaborate with for social media campaigns",
                   "Generate a script for a podcast episode discussing industry trends",
                   "Write a script for a webinar on best practices for using our product",
                   "Create a list of potential case studies to showcase our company’s success",
                   "Generate a script for a short video on our company’s history and achievements",
                   "Write a script for a virtual event to launch our new product",
                   "Create a list of potential topics for a company newsletter",
                   "Generate a script for a TV commercial to increase brand awareness",
                   "Write a script for an explainer video about our company’s sustainability efforts",
                   "Can you think of new business ideas without money?",
                   "Write a persuasive email to increase attendance at our upcoming event",
                   "Create a follow-up email to send to potential clients after a meeting",
                   "Generate a thank-you email to send to customers after a purchase",
                   "Write a promotional email to introduce our new product or service",
                   "Create a reminder email for an upcoming deadline or meeting",
                   "Generate a professional email to request a meeting or consultation",
                   "Write an apology email to a customer for a delay or mistake",
                   "Create a personalized email to nurture a lead and move them closer to a sale",
                   "Generate an email to request a referral or testimonial from a satisfied customer",
                   "Write a promotional email to announce a sale or special offer",
                   "Create an email to send to a prospect who has shown interest in our product",
                   "Generate an email to request feedback from customers on our product or service",
                   "Write an email to send to a customer who has unsubscribed from our mailing list",
                   "Create an email to send to a potential partner to explore collaboration opportunities",
                   "Generate an email to send to a customer with a personalized upselling or cross-selling suggestion",
                   "Write a daily to-do list for the sales team",
                   "Generate a daily summary of customer feedback and testimonials",
                   "Write a daily agenda for the executive team meeting",
                   "Write a persuasive email to convince potential customers to try our service"]

        self.__current_inputs = self.__tokenize_func(random.choice(prompts))
        self.__current_inputs = {key: value for key, value in self.__current_inputs.items()}

        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        encoded_input = tokenizer('hi, how are you?', return_tensors='pt')
        input_dict = {key: value for key, value in encoded_input.items()}

        # return self.__current_inputs['input_ids']
        return input_dict

    def submit_count(self, batch_size):
        self._idx += batch_size

    def reset(self):
        self._idx = 0
        return True

    def summarize_accuracy(self):
        print("accuracy metrics for this model are under development")
        return {}
