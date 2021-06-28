import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession('onnx/u2net.onnx')

print(len(ort_session.get_inputs()))
print(len(ort_session.get_outputs()))

input_name= ort_session.get_inputs()[0].name

loc1_name = ort_session.get_outputs()[0].name
loc2_name = ort_session.get_outputs()[1].name
loc3_name = ort_session.get_outputs()[2].name
loc4_name = ort_session.get_outputs()[3].name
loc5_name = ort_session.get_outputs()[4].name
loc6_name = ort_session.get_outputs()[5].name
loc7_name = ort_session.get_outputs()[6].name


print(input_name)
print(loc1_name)
print(loc2_name)
print(loc3_name)
print(loc4_name)
print(loc5_name)
print(loc6_name)
print(loc7_name)

outputs = ort_session.run([loc1_name,loc2_name,loc3_name,loc4_name,loc5_name,loc6_name,loc7_name], {input_name:np.random.randn(1, 3, 320, 320).astype(np.float32)})


print(outputs[0].shape)
print(outputs[1].shape)
print(outputs[2].shape)
print(outputs[3].shape)
print(outputs[4].shape)
print(outputs[5].shape)
print(outputs[6].shape)
#print(outputs.size)