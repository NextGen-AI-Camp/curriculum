# Table of Content

Week 1:<br />
&nbsp;&nbsp;&nbsp;&nbsp;[Python #1](#python-1)<br />
&nbsp;&nbsp;&nbsp;&nbsp;[การใช้งาน Google Colab](#การใช้งาน-google-colab)<br />
&nbsp;&nbsp;&nbsp;&nbsp;[Python #2](#python-2)<br />
&nbsp;&nbsp;&nbsp;&nbsp;[Introduction to AI, ML, DL](#introduction-to-ai-ml-dl)<br />

Week 2:<br />
&nbsp;&nbsp;&nbsp;&nbsp;[Python #3](#python-3)<br />
&nbsp;&nbsp;&nbsp;&nbsp;[Neural Network](#neural-network)<br />
&nbsp;&nbsp;&nbsp;&nbsp;[NumPy #1](#numpy-1)<br />
&nbsp;&nbsp;&nbsp;&nbsp;[Matplotlib](#matplotlib)<br />

Week 3:<br />
&nbsp;&nbsp;&nbsp;&nbsp;[Neural Network #2](#neural-network-2)<br />
&nbsp;&nbsp;&nbsp;&nbsp;[NumPy #2](#numpy-2)<br />
&nbsp;&nbsp;&nbsp;&nbsp;[Pandas](#pandas)<br />
&nbsp;&nbsp;&nbsp;&nbsp;[Pytorch #1](#pytorch-1)<br />

Week 4:<br />
&nbsp;&nbsp;&nbsp;&nbsp;[Neural Network #3](#neural-network-3)<br />
&nbsp;&nbsp;&nbsp;&nbsp;[OpenCV #1](#opencv-1)<br />
&nbsp;&nbsp;&nbsp;&nbsp;[OpenCV #2](#opencv-2)<br />

Week 5:<br />
&nbsp;&nbsp;&nbsp;&nbsp;[CNN #1](#cnn-1)<br />
&nbsp;&nbsp;&nbsp;&nbsp;[CNN #2](#cnn-2)<br />
&nbsp;&nbsp;&nbsp;&nbsp;[TorchVision](#torchvision)<br />

Week 6:<br />
&nbsp;&nbsp;&nbsp;&nbsp;[CNN Model and Transfer Learning](#cnn-model-and-transfer-learning)<br />

---

## Week 1:

### Python #1 

ในหัวข้อนี้ จะเริ่มต้นด้วยการติดตั้ง Python และการตั้งค่าในระบบต่าง ๆ ต่อมาจะเรียนรู้เกี่ยวกับการรับและแสดงผลข้อมูล (I/O) รวมถึงการทำงานกับประเภทข้อมูลพื้นฐาน (datatype) และตัวแปร (variables) เพื่อจัดเก็บข้อมูล นอกจากนี้ยังจะครอบคลุมการดำเนินการทางคณิตศาสตร์ (math operations) เบื้องต้น ซึ่งเป็นพื้นฐานที่สำคัญในการเขียนโปรแกรมและการแก้ปัญหาทางคอมพิวเตอร์

- Video: [NextGenAI-2024 | Python #1](https://youtu.be/Yizq4I6JThY?si=vOa8_ghIGK5pN9b2)
- Code: [NextGen_AI_Camp_Python#1.ipynb](https://github.com/NextGen-AI-Camp/curriculum/blob/main/Week%231/Python%231/NextGen_AI_Camp_Python%231.ipynb)

### การใช้งาน Google Colab
ในหัวข้อนี้ เราจะเรียนรู้การใช้งาน Google Colab ซึ่งเป็นเครื่องมือสำหรับการเขียนและรันโค้ด Python บนคลาวด์ ผู้เรียนจะได้เรียนรู้วิธีการตั้งค่าและใช้ทรัพยากรจาก Google Colab เช่น การเลือกใช้ CPU หรือ GPU เพื่อเพิ่มประสิทธิภาพการประมวลผล นอกจากนี้ยังจะได้เรียนรู้เกี่ยวกับการใช้ Jupyter Notebook ซึ่งเป็นเครื่องมือที่สำคัญสำหรับนักพัฒนาและนักวิจัย
- Video: [NextGenAI-2024 | Google Colab](https://youtu.be/znOQg9Ax42Q?si=IcBDYol6IHL1gVri)
- Code: [NextGen_AI_Camp_Google_Colab.ipynb](https://github.com/NextGen-AI-Camp/curriculum/blob/main/Week%231/Google_Colab/NextGen_AI_Camp_Google_Colab.ipynb)

### Python #2
ในหัวข้อนี้ เราจะเน้นการเขียนโค้ดที่มีเงื่อนไข (conditions) ซึ่งเป็นส่วนสำคัญในการตัดสินใจของโปรแกรม รวมถึงการใช้ตัวดำเนินการตรรกศาสตร์ (logical operators) สำหรับการตรวจสอบเงื่อนไขต่าง ๆ นอกจากนี้ยังจะได้เรียนรู้เกี่ยวกับการใช้ลิสต์ (lists) สำหรับการจัดเก็บและจัดการกลุ่มข้อมูล รวมถึงการจัดการกับสตริง (strings) ซึ่งเป็นข้อมูลประเภทข้อความที่ใช้บ่อยในโปรแกรม
- Video: [NextGenAI-2024 | Python #2](https://youtu.be/7SmFEwKcbTA)
- Code: [NextGen_AI_Camp_Python#2.ipynb](https://github.com/NextGen-AI-Camp/curriculum/blob/main/Week%231/Python%232/NextGen_AI_Camp_Python%232.ipynb)

### Introduction to AI, ML, DL
ในหัวข้อนี้ เราจะเริ่มต้นด้วยการแนะนำเกี่ยวกับปัญญาประดิษฐ์ (AI) ซึ่งเป็นการพัฒนาระบบคอมพิวเตอร์ให้สามารถทำงานที่ต้องใช้ความฉลาดของมนุษย์ นอกจากนี้ยังเรียนรู้เรื่องของ Machine Learning ซึ่งเป็นการใช้ข้อมูลและอัลกอริธึมเพื่อให้คอมพิวเตอร์เรียนรู้จากประสบการณ์ รวมถึง Deep Learning ซึ่งเป็นการสร้างโมเดลที่ซับซ้อนและมีความสามารถในการรับรู้และประมวลผลข้อมูลที่ซับซ้อนยิ่งขึ้น จะได้เห็นถึงความเหมือนและความต่างของทั้งสามแนวคิดนี้ รวมถึงแอพพลิเคชันที่นำไปใช้ในโลกจริง

- Video: [COMING SOON]
- Code: [COMING SOON]
