def get_centre_contexts(text, window_size=5):
    centre_contexts = []
    for i in range(window_size, len(text) - window_size):
        centre = text[i]
        context = text[i - window_size : i] + text[i + 1 : i + window_size + 1]
        centre_contexts.append((centre, context))
    return centre_contexts
