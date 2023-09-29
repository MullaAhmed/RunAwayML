import json
from bs4 import BeautifulSoup
from jinja2 import Template, FileSystemLoader, Environment,Template
from functions.content_functions import *
from functions.images_functions import *
from functions.constants import *

IMAGE_FUNCTIONS={
    "Stable Diffusion":generate_sd_image,
    "Unsplash":get_unsplash_img,
    "Pexel":get_pexels_img
}


def generate_content(slides,topic,engine,api_key=None,use_chat=True):

  data_dict = json.load(open('./templates/config.json'))
  json_data={}
  for slide in slides:
      json_data[slide]=data_dict[slide.split(":")[1]]

  prompt= """fill this json with content regarding the {topic} title must be less than 7 words, subtitle must be less than 17 words , paragraphs must an array, each of atleast 120 words, and points must be an array of at most 5 each of atleast 10 words, img must be a text description of an image corresponding to the content of the slide
          {json_format}""".format(topic=topic,json_format=json_data)
  
  content=None
  if api_key!=None:
      while content==None:
          try:
              content=get_content_from_openai(prompt,engine,api_key,use_chat)
          except Exception as e:
              content=None
              print("Error: ",e)
  try:
      dict_content=json.loads(content)
      return dict_content

  except:
    generate_content(slides,topic,engine,api_key,use_chat)

def generate_template(style,slides,content,bg_image,image_source="Unsplash",model= "Lykon/DreamShaper"):

  loader = FileSystemLoader('./templates/components')
  env = Environment(loader=loader)

  # Load the template file
  style = open("./templates/styles/"+style).read()
  components_html = open('./templates/components/components.html').read()

  template = env.get_template('template.html')

  # Render the template
  style = style.replace("#bg_img",bg_image)

  components=[]
  for slide in slides:
    
    soup = BeautifulSoup(components_html, 'html.parser')
    selected_div = soup.find('div', class_=slide.split(':')[1])
    if slide.split(":")[1]=='img':
      orientation='landscape'
    else:
      orientation='portrait'

    if 'img' in content[slide].keys():
      try:
        img_prompt=content[slide]['img']
        img=IMAGE_FUNCTIONS[image_source](img_prompt, orientation,None,model)
        content[slide]['img']=image_to_base64(img)
      except Exception as e:
        img_prompt=content[slide]['img']
        img=generate_sd_image(img_prompt, orientation,None,model)
        content[slide]['img']=image_to_base64(img)
        print("Error: ",e)
    div_jinja=Template(str(selected_div))
    
    selected_div=div_jinja.render(content[slide])
    components.append(selected_div)

  data = {
      'style': style,
      'components': components
  }

  rendered_html_template = Template(template.render(data))

  return rendered_html_template

