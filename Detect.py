import streamlit as st
import tensorflow as tf
import tf_keras

st.write("# Animal Detection")
file = st.file_uploader("Upload an image")
class_names = ['Bear', 'Brown bear', 'Bull', 'Butterfly', 'Camel', 'Canary', 'Caterpillar',
                'Cat', 'Cattle', 'Centipede', 'Cheetah', 'Chicken', 'Crab', 'Crocodile', 'Deer',
                'Dog', 'Duck', 'Eagle', 'Elephant', 'Fish', 'Fox', 'Frog', 'Giraffe', 'Goat',
                'Goldfish', 'Goose', 'Hamster', 'Harbor seal', 'Hedgehog', 'Hippopotamus',
                'Horse', 'Jaguar', 'Jellyfish', 'Kangaroo', 'Koala', 'Ladybug', 'Leopard',
                'Lion', 'Lizard', 'Lynx', 'Magpie', 'Monkey', 'Moths and butterflies', 'Mouse',
                'Mule', 'Ostrich', 'Otter', 'Owl', 'Panda', 'Parrot', 'Penguin', 'Pig',
                'Polar bear', 'Rabbit', 'Raccoon', 'Raven', 'Red panda', 'Rhinoceros',
                'Scorpion', 'Sea lion', 'Sea turtle', 'Seahorse', 'Shark', 'Sheep', 'Shrimp',
                'Snail', 'Snake', 'Sparrow', 'Spider', 'Squid', 'Squirrel', 'Starfish', 'Swan',
                'Tick', 'Tiger', 'Tortoise', 'Turkey', 'Turtle', 'Whale', 'Woodpecker', 'Worm',
                'Zebra']

if file is not None:
    if file.name.split(".")[-1] in ['jpg', 'jpeg', 'png', "jfif"]:
        # Read in the image
        img = file.read()
        st.image(img, width=600)
        # Decode it into a tensor
        img = tf.image.decode_jpeg(img)
        # Resize the image
        img = tf.image.resize(img, [224, 224])

        model = tf_keras.models.load_model('animal_detection')
        img=img/255.
        result = model.predict(tf.expand_dims(img, axis=0))
        pred_class = class_names[result.argmax()]
        button = st.button("Predict")
        if button:
            st.write(f"Prediction: {pred_class} ({round(result[0][result.argmax()]*100,2)}%)")