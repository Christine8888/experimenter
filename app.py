import anthropic 
import yaml
from typing import List, Dict, Tuple
from vector_store import EmbeddingClient, Document, DocumentLoader
import semantic_search
import hyde
import gradio as gr
import os
import hyde_reranking
import tempfile
import json
from openai import OpenAI
from sciencetree import scienceTreeNode

# retriever = hyde_reranking.HydeCohereRetrievalSystem()
retriever = semantic_search.EmbeddingRetrievalSystem()

def generate_science_tree(science_objective, openai_api_key, anthropic_api_key, branching_factor, year_cutoff, temperature):
    print(f"Received inputs: {science_objective}, {openai_api_key}, {anthropic_api_key}, {branching_factor}, {year_cutoff}, {temperature}")
    
    # set API keys from user
    try:
        retriever.set_clients(openai_api_key, anthropic_api_key)
    except Exception as e:
        print(e)
        print("Unable to connect to OpenAI API.")
    
    tree = scienceTreeNode(text=science_objective, year=year_cutoff, retriever=retriever, n=branching_factor, 
    temperature=temperature, generation_model = "claude-3-5-sonnet-20240620", api_key = anthropic_api_key)

    
    def build_tree_structure(node):
        result = {
            "text": node.text,
            "papers": [format_paper_link(paper) for paper in (node.docs if hasattr(node, 'docs') else [])]
        }
        if node.children:
            result["children"] = [build_tree_structure(child) for child in node.children]
        return result

    # convert to dictionary
    tree_structure = build_tree_structure(tree)

    return tree_structure

def format_paper_link(paper_id):
    if '.txt' in paper_id:
        arxiv_id = paper_id.split('_')[0].split('astro-ph')[-1]
        return f"https://arxiv.org/abs/astro-ph/{arxiv_id}"
    else:
        return f"https://arxiv.org/abs/{paper_id}"

def save_tree_to_file(tree):
    if tree is None:
        print("No tree to save.")
        return None
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
        json.dump(tree, temp_file)
    return temp_file.name

def load_tree_from_file(file):
    if file is None:
        return None
    with open(file.name, 'r') as f:
        return json.load(f)

def create_tree_html(tree):
    if tree is None:
        return ""
    
    html = "<ul>"
    html += "<li>"
    
    # Create dropdown view
    html += f"<details><summary>{tree['text']}</summary>"
    
    if tree['papers']:
        html += "<p>"
        paper_links = [f"<a href='{paper}' target='_blank'>{paper.split('/')[-1]}</a>" for paper in tree['papers']]
        html += ", ".join(paper_links)
        html += "</p>"
    
    
    if 'children' in tree and tree['children']:
        html += "<ul>"
        for child in tree['children']:
            html += create_tree_html(child)
        html += "</ul>"
    
    
    html += "</details>"
    
    html += "</li></ul>"
    return html
css = """
uul, details, summary, p {
    font-size: 1em; /* Set a uniform font size for all elements */
    margin: 0;
    padding: 0;
}

ul {
    list-style-type: none;
    padding-left: 20px;
}

details summary {
    cursor: pointer;
    font-weight: bold;
    margin-bottom: 5px;
    padding-left: 10px;
}

details[open] > summary::after {
    content: '▲';  /* Use an up arrow when expanded */
    float: right;
    margin-left: 10px;
}

details > summary::after {
    content: '▼';  /* Use a down arrow when collapsed */
    float: right;
    margin-left: 10px;
}

p {
    margin-top: 5px;
    margin-bottom: 5px;
}


.file-custom {
    height: 80px;
}
"""


with gr.Blocks(css=css) as iface:
    gr.Markdown("# Experiment Generator")
    gr.Markdown("Generate sample experiments derived on top-level science goals and scientific literature search.")

    with gr.Row():
        with gr.Column(scale=1):
            science_objective = gr.Textbox(label="Science Goal")
            branching_factor = gr.Slider(label="Branching Factor", minimum=1, maximum=10, step=1, value=2)
            temperature = gr.Slider(label="Temperature", minimum=0, maximum=1, step=0.1, value=0.5)
            year_cutoff = gr.Slider(label="Year Cutoff", minimum=2000, maximum=2024, step=1, value=2024)
            openai_key = gr.Textbox(label="OpenAI API Key", type="password")
            anthropic_key = gr.Textbox(label="Anthropic API Key", type="password")

            submit_btn = gr.Button("Generate")

        with gr.Column(scale=2):
            tree_output = gr.JSON(label="Tree Data", visible=False)
            tree_display = gr.HTML(label="Tree Display")

            with gr.Row():
                save_btn = gr.Button("Save Tree", size="sm")
                load_btn = gr.Button("Load Tree", size="sm")

            file_output = gr.File(label="Tree File", elem_classes=["file-custom"])
            file_input = gr.File(label="Load Tree File", elem_classes=["file-custom"])

    def update_tree_display(tree):
        if tree is not None:
            return create_tree_html(tree)
        return gr.update()

    submit_btn.click(
        generate_science_tree,
        inputs=[science_objective, openai_key, anthropic_key, branching_factor, year_cutoff, temperature],
        outputs=[tree_output]
    ).then(
        update_tree_display,
        inputs=[tree_output],
        outputs=[tree_display]
    )

    save_btn.click(
        save_tree_to_file,
        inputs=[tree_output],
        outputs=[file_output]
    )

    file_input.change(
        load_tree_from_file,
        inputs=[file_input],
        outputs=[tree_output]
    ).then(
        update_tree_display,
        inputs=[tree_output],
        outputs=[tree_display]
    )


iface.launch()