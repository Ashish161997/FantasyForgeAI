import os
from crewai import Agent, Task, Crew, Process, LLM
import gradio as gr
from markdown import markdown
import warnings
warnings.filterwarnings('ignore')

api_key = os.getenv("api_key")
llm = LLM(
              model='gemini/gemini-1.5-flash',
              api_key=api_key
            )


# Agents
def generate_fantasy_outline(user_idea:str):
    idea_extractor = Agent(
        role="Idea Extractor",
        goal=f"Extract and refine a raw fantasy story idea into structured elements from {user_idea}.",
        backstory="You are a creative consultant helping fantasy writers refine their initial story ideas. "
                  "You specialize in identifying characters, setting, tone, conflict, and themes. "
                  "You guide the entire outlining process by establishing a clear vision.",
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )
    
    worldbuilder = Agent(
        role="Worldbuilding Expert",
        goal="Design an ULTRA-DETAILED fantasy world with comprehensive systems.",
        backstory="You are a seasoned worldbuilder who constructs vivid and believable fantasy realms. "
                  "You define geography, cultures, magic systems, political structures, and mythologies "
                  "that serve as the foundation for storytelling.",
        llm=llm,
        allow_delegation=False,
        verbose=True,

    )
    
    character_architect = Agent(
        role="Character Architect",
        goal="Create detailed, multi-dimensional characters who drive the story forward.",
        backstory="You are a character development expert who crafts heroes, villains, and side characters. "
                  "You define their backstories, motivations, arcs, relationships, and emotional journeys.",
        llm=llm,
        allow_delegation=False,
        verbose=True,
       
    )
    
    plot_strategist = Agent(
        role="Plot Structure Strategist",
        goal="Generate a structured fantasy plot using traditional narrative frameworks.",
        backstory="You are a master storyteller who weaves plots using the Hero’s Journey, Three Act Structure, and other frameworks. "
                  "You ensure that the plot flows logically and engages readers from start to finish.",
        llm=llm,
        allow_delegation=False,
        verbose=True,
       
    )
    
    scene_director = Agent(
        role="Scene Breakdown Specialist",
        goal="Translate the structured plot into detailed scene-by-scene or chapter-wise outlines.",
        backstory="You specialize in breaking stories into vivid scenes. "
                  "You provide clear settings, goals, conflicts, and turning points for each scene "
                  "to help the writer visualize and write the novel more easily.",
        llm=llm,
        allow_delegation=False,
        verbose=True,
     
    )
    
    theme_advisor = Agent(
        role="Theme & Tone Advisor",
        goal="Ensure thematic consistency and emotional depth across the outline align with {user_idea}.",
        backstory="You are a literary expert who analyzes and suggests core themes, tones, and motifs. "
                  "You help the story feel emotionally resonant and stylistically consistent throughout.",
        llm=llm,
        allow_delegation=False,
        verbose=True,
      
    )
    
    consistency_checker = Agent(
        role="Continuity & Logic Checker",
        goal="Identify logical gaps, plot holes, and inconsistencies in the full outline.",
        backstory="You are an editorial specialist who reviews the full outline for quality control. "
                  "You ensure that the plot, characters, and world are coherent and nothing is left underdeveloped.",
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )
    
    
    #tasks
    
    
    idea_extraction_task = Task(
    description=(
        f"1. Analyze the user's raw fantasy novel idea: {user_idea}.\n"
        "2. Identify the main characters, potential setting, central conflict, and themes.\n"
        "3. Clarify vague elements and structure the idea into a coherent story seed.\n"
        "4. Prepare the refined concept for use in worldbuilding and plotting.\n"
        "5. Generate 1000+ words with exhaustive commentary"
    ),
    expected_output="A structured 1000+ word summary of the story idea including protagonist, world hints, tone, conflict, and themes.",
    agent=idea_extractor
    )
    
    worldbuilding_task = Task(
        description=(
            "1. Use the refined idea to design a vivid and immersive fantasy world.\n"
            "2. Define the geography, magic systems, political structures, and cultural norms.\n"
            "3. Include religion, history, economic systems, and mythologies where relevant.\n"
            "4. Ensure internal consistency and creative depth across the world design.\n"
            "5.Write at least 1000 words."
        ),
        expected_output="A detailed 2000+ word world document covering terrain, magic, politics, culture, and unique lore.",
        agent=worldbuilder
    )
    
    character_task = Task(
        description=(
            "1. Create detailed character profiles for at least one protagonist, one antagonist, and two key supporting characters.\n"
            "2. Define names, roles, traits, motivations, arcs, and relationships.\n"
            "3. Ensure each character supports the core theme and plot direction.\n"
            "4. Highlight how their evolution affects the story.\n"
            "5. Write at least 800+ words."
        ),
        expected_output="Four or more character bios totaling 800+ words with backstory, arc, and inter-character dynamics.",
        agent=character_architect
    )
    
    plot_task = Task(
        description=(
            "1. Develop a plot outline using the Hero’s Journey or Three-Act Structure.\n"
            "2. Include exposition, inciting incident, rising action, climax, and resolution.\n"
            "3. Integrate major turning points, conflicts, and internal dilemmas.\n"
            "4. Align plot beats with character and theme development.\n"
            "5. Write at least 800 words."
        ),
        expected_output="A structured plot outline with 800+ words covering 8–12 major beats linked to characters and world.",
        agent=plot_strategist
    )
    
    scene_task = Task(
        description=(
            "1. Break the structured plot into a detailed scene-by-scene or chapter-wise outline.\n"
            "2. For each scene, define the setting, involved characters, scene goal, and conflict.\n"
            "3. Include pacing considerations and emotional flow.\n"
            "4. Ensure each scene advances the plot and develops theme or character.\n"
            "5. Write at least 1000+ words."
        ),
        expected_output="A comprehensive 1000+ word scene-by-scene breakdown to guide novel structure.",
        agent=scene_director
    )
     
    theme_task = Task(
        description=(
            "1. Analyze the story concept and outline to extract central themes and motifs.\n"
            "2. Identify emotional tone (e.g., epic, tragic, whimsical) and stylistic cues.\n"
            "3. Suggest recurring symbols or metaphors that unify the narrative.\n"
            "4. Ensure tonal consistency from beginning to end.\n"
            "5. Write at least 400+ words."
        ),
        expected_output="A 400+ word theme and tone guideline covering core message, style, and motifs.",
        agent=theme_advisor
    )
    
    consistency_task = Task(
        description=(
            "1. Review the complete outline for logic gaps, inconsistencies, or contradictions.\n"
            "2. Identify undeveloped or repetitive sections.\n"
            "3. Ensure character motivations, world rules, and plot developments align.\n"
            "4. Suggest direct edits or improvements for cohesion.\n"
            "5. Write at least 2000+ words."
        ),
        expected_output="A 2000+ word edited and annotated version of the full outline with feedback and fixes.",
        agent=consistency_checker
    )
  
    
    # Assemble all the tasks
    tasks = [
        idea_extraction_task,
        worldbuilding_task,
        character_task,
        plot_task,
        scene_task,
        theme_task,
        consistency_task
    ]
    
    # Define the crew with all agents and tasks
    fantasy_outline_crew = Crew(
        agents=[
            idea_extractor,
            worldbuilder,
            character_architect,
            plot_strategist,
            scene_director,
            theme_advisor,
            consistency_checker
        ],
        tasks=tasks,
        verbose=True
    )

    results=fantasy_outline_crew.kickoff(inputs={'user_idea':user_idea})
    return results.raw



def format_output(plan_text):
    # Convert plain text or Markdown to styled HTML
    html_content = markdown(plan_text)
    styled_html = f"""
    <div style='
        font-family: "Segoe UI", sans-serif;
        line-height: 1.6;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #ccc;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        color: #333;
    '>
        {html_content}
    </div>
    """
    return styled_html

import base64

# Function to encode the image
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Your image path - REPLACE THIS WITH YOUR ACTUAL IMAGE PATH
IMAGE_PATH = "fantasy.jpeg"  # or "C:/path/to/your/background.jpg"

# Get base64 encoded image
try:
    encoded_image = get_base64_image(IMAGE_PATH)
except FileNotFoundError:
    print(f"Error: Image not found at {IMAGE_PATH}")
    encoded_image = ""

CSS = f"""
.gradio-container {{
    background: url('data:image/jpeg;base64,{encoded_image}') !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    min-height: 100vh;
    padding: 30px;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}}
.main-title {{
    text-align: center;
    font-size: 72px !important;
    font-weight: 800;
    color: #ffffff !important;
    margin-bottom: 15px;
    text-shadow: 3px 3px 12px #000000;
    padding: 30px;
    background: rgba(20, 20, 40, 0.7);
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 3px;
    border: 2px solid rgba(255,255,255,0.2);
}}
.sub-title {{
    text-align: center;
    font-size: 22px;
    color: #ffffff;
    margin-bottom: 40px;
    text-shadow: 2px 2px 6px #000000;
    background: rgba(0, 0, 0, 0.6);
    padding: 15px;
    border-radius: 10px;
    max-width: 80%;
    margin-left: auto;
    margin-right: auto;
}}
.gradio-button {{
    background-color: #4CAF50 !important;
    color: white !important;
    font-size: 18px !important;
    padding: 16px 24px !important;
    border-radius: 8px !important;
    margin-top: 20px !important;
    border: none !important;
    transition: all 0.3s ease !important;
}}
.gradio-button:hover {{
    background-color: #3e8e41 !important;
    transform: scale(1.05) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
}}
#output-box {{
    height: 600px;
    overflow-y: scroll;
    padding: 25px;
    background-color: rgba(255, 255, 255, 0.92) !important;
    border: 1px solid #cccccc !important;
    border-radius: 15px !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.15) !important;
}}
.textbox {{
    background: rgba(255,255,255,0.9) !important;
}}
"""



def format_output(plan_text):
    # Convert plain text or Markdown to styled HTML
    html_content = markdown(plan_text)
    styled_html = f"""
    <div style='
        font-family: "Segoe UI", sans-serif;
        line-height: 1.6;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #ccc;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        color: #333;
    '>
        {html_content}
    </div>
    """
    return styled_html

def create_gradio_app():
    with gr.Blocks(css=CSS) as demo:
        # gr.Markdown("""<div class='main-title'>FANTASY NOVEL OUTLINE GENERATOR</div>""")
        # gr.Markdown("""<div class='sub-title'>Transform your creative vision into a fully-developed fantasy world</div>""")

        with gr.Row():
            with gr.Column(scale=1):
                user_idea = gr.Textbox(
                    label="Enter Your Fantasy Novel Idea",
                    placeholder="A hidden prophecy... a forgotten magic... an unlikely hero...",
                    lines=5,
                    interactive=True
                )
                generate_btn = gr.Button("Craft My Outline", elem_classes="gradio-button")

            with gr.Column(scale=2):
                output_html = gr.HTML(label="Your Generated Outline",elem_id="output-box")

        generate_btn.click(
            fn=lambda idea: format_output(generate_fantasy_outline(idea)),
            inputs=[user_idea],
            outputs=output_html
        )

    return demo

if __name__ == "__main__":
    app=create_gradio_app()
    app.launch()
