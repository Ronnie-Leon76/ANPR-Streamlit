import streamlit as st
import PIL
import cv2
import numpy as np
import imutils
import easyocr
import tempfile


# Setting page layout
st.set_page_config(
    page_title="Automatic Number Plate License Detection",  # Setting page title
    page_icon="🚗",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default   
)

# Creating sidebar
with st.sidebar:
    st.header("Image Config")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    

# Creating main page heading
st.title("Automatic Number Plate License Detection")
st.caption('Upload an image of a vehicle with a number plate.')
st.caption('Then click the :blue[Detect License Plate] button and check the result.')
# Creating two columns on the main page
col1, col2 = st.columns(2)


# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        print(uploaded_image)
        # Adding the uploaded image to the page with a caption
        st.image(source_img,
                 caption="Uploaded Image",
                 use_column_width=True
                 )
        

if st.sidebar.button('Detect License Plate'):
    # Save the uploaded image to a temporary file and read it
    tfile = tempfile.NamedTemporaryFile(delete=True)
    tfile.write(source_img.read())

    # Read image
    img = cv2.imread(tfile.name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply filter and find edges for localization
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
    edged = cv2.Canny(bfilter, 30, 200) #Edge detection

    # Find contours and apply mask
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0,255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)


    # Crop license plate
    (x,y) = np.where(mask==255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped_image = gray[topx:bottomx+1, topy:bottomy+1]


    # Use Easy OCR to read text
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)

    with col2:
        try:
            text = result[0][-2]
        except Exception as e:
            text = "No Text Detected"
        font = cv2.FONT_HERSHEY_SIMPLEX
        res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
        res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
        st.image(cv2.cvtColor(res, cv2.COLOR_BGR2RGB), caption="Detected License Plate", use_column_width=True)

        try:
            st.write("Detected License Plate:", text)
        except Exception as e:
            st.write("No License Plate Detected")