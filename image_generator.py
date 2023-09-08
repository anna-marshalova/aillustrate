from utils import cleanup

class ImageGenerator:
    def __init__(self, pipe, device, neg_prompt = None):
        self.pipe = pipe
        self.device = device
        self.neg_prompt = neg_prompt
        self.REGENERATE_STEPS = 5
        self.pipe.to(self.device)

    def generate(self, prompt):
        i = 0
        nsfw_content_detected = True
        while i < self.REGENERATE_STEPS and nsfw_content_detected:
            output = self.pipe(prompt, negative_prompt = self.neg_prompt)
            image = output.images[0]
            nsfw_content_detected = output.nsfw_content_detected[0]
            i += 1
        cleanup()
        return image