from flask import Flask, jsonify, request, send_file
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from functions.main_functions import *
from PIL import Image
from functions.constants import *

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'secret-key'  # Replace with your own secret key
jwt = JWTManager(app)

# Endpoint for user login
@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')

    # Verify username and password (You can replace this with your own authentication logic)
    if username == 'admin' and password == 'password':
        # Create access token with user identity
        access_token = create_access_token(identity=username)
        return jsonify({'access_token': access_token}), 200
    else:
        return jsonify({'message': 'Invalid credentials'}), 401

# Protected endpoint that requires JWT authentication
@app.route('/app', methods=['POST'])
@jwt_required()
def generator():
    # Get the user identity from the JWT
    current_user = get_jwt_identity()

    topic=request.json.get('topic')
    style =request.json.get('style')
    slides = request.json.get('slides')
    image_source= request.json.get('image_source')
    sd_model= request.json.get('sd_model')
    engine=request.json.get('engine')
    api_key=request.json.get('api_key')
    use_chat=bool(request.json.get('use_chat'))


    print(topic,style,slides,use_chat)
    bg_image=image_to_base64(Image.open(BACKGROUNDS[style]), "png")
    content= generate_content(slides,topic,engine,api_key,use_chat)
    # content=json.load(open('c.json'))
    print("done with content")
    html=generate_template(style, slides,content,bg_image,image_source,sd_model).render()
    render_pdfkit(html,f"{current_user}.pdf")

    return send_file(path_or_file=f"{current_user}.pdf", as_attachment=True, download_name=f"{current_user}.pdf")

    # return html

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")
