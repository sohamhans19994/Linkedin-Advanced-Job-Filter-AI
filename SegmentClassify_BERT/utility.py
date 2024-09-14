from bs4 import BeautifulSoup

def split_sections(job_desc_html):
    soup = BeautifulSoup(job_desc_html, 'html.parser')
    topics = []
    current_topic = {"title": "First Overview", "content": ""}
    first_topic_added = False
    
    # Process each <span> block
    for span in soup.find_all('span'):
        # Check if the span contains a <strong> element
        strong = span.find('strong')
        
        if strong:
            # If the current topic has content, finalize it
            if current_topic["content"].strip() or not first_topic_added:
                topics.append(current_topic)
                first_topic_added = True
            # Start a new topic
            current_topic = {"title": strong.get_text(strip=True), "content": ""}
        else:
            # Append content to the current topic
            paragraphs = span.find_all('p')
            for p in paragraphs:
                current_topic["content"] += p.get_text(separator='\n', strip=True) + "\n"
            
            # Append list items if any
            lists = span.find_all('ul')
            for ul in lists:
                list_items = "\n".join(li.get_text(separator='\n', strip=True) for li in ul.find_all('li'))
                current_topic["content"] += list_items + "\n"

    # Add the last topic if it exists and has content
    if current_topic and (current_topic["content"].strip() or not first_topic_added):
        topics.append(current_topic)

    topic_text_blocks = []
    for topic in topics:
        topic_text_blocks.append(topic["title"] + "\n" + topic["content"])
    return topic_text_blocks

