image_size = 1024
uploaded_image = load_image(uploaded_file)

###########
uploaded_image_my = uploaded_image.copy()
uploaded_image_my_as_rgb = cv2.cvtColor(uploaded_image_my, cv2.COLOR_BGR2RGB) #1
model_my = load_model_my()

###########

uploaded_image = cv2.resize(uploaded_image, (image_size, image_size))
uploaded_image_as_rgb = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB) #1

model = load_model(image_size=image_size)
x = preprocess_input(uploaded_image_as_rgb) #2
x_batched = np.expand_dims(x, axis=0)
predicted = model(x_batched) #3
heatmap = make_heatmap(x_batched, model, 'block14_sepconv2_act') #4
merged = merge_heatmap(uploaded_image_as_rgb, heatmap) #5




#######################################
h_img = uploaded_image_my.shape[0]
w_img = uploaded_image_my.shape[1]
x_my = preprocess_image_custom(uploaded_image_my) #2

original_images = []
img_tiles = []
heatmaps_my = []
original_images_as_rgb = []
merged_list = []

for i in range(h_img//224):
    for j in range(w_img//224):
        cropped = x_my[0][i*224:i*224+224 , j*224:j*224+224 ,:]
        ori = uploaded_image_my[i*224:i*224+224 , j*224:j*224+224 ,:]
        img_tiles.append(cropped)
        original_images.append(ori)
        
predicted_my = model_my(np.array(img_tiles)) #3

for img in img_tiles:
    heatmap_now = make_heatmap(np.array([img]), model_my, 'post_relu') #4
    heatmaps_my.append(heatmap_now)

for ori_img in original_images:
    ori_img_as_rgb = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB) #1
    original_images_as_rgb.append(ori_img_as_rgb)

for i in range(len(original_images_as_rgb)):
    merged_now = merge_heatmap(original_images_as_rgb[i], heatmaps_my[i]) #5
    merged_list.append(merged_now)



columns = st.beta_columns(2)

with columns[0]:
    st.header('Uploaded Image')
    st.image(original_images[1], channels='BGR')
    
    
with columns[1]:
    st.header('Result')
    st.image(merged_list[1])
