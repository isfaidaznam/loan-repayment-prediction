def print_title(text):
    line_length = 60
    text = str(text).strip().title()
    if len(text) > line_length + 1:
        line_length = len(text) + 1
    title = "\n" + ("="*line_length) + "\n" + str(text).strip().title() + "\n" + ("="*line_length)
    print(title)
    pass