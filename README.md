# Text-Detection-Recognition

# setup
If you want to run evaluation on COCO-Text dataset, download and unizip [train2014.zip](http://images.cocodataset.org/zips/train2014.zip) to ./COCO directory.  
Then run the following script.
```shell
python eval_on_coco_text.py
```

# demo
If you just want ocr on images, run ctpn_crnn_pytorch.py.  
PyTorch version of CRNN yields much better result than Tensorflow version.
```shell
python ctpn_crnn_pytorch.py
```

# report
For implementation detail or experiment results, refer to the [report](20180624_Text-Detection-Recognition.pdf).


# some results
<img src="/ocr_result/20180415_110144.jpg" width=501 height=282/>
<img src="/ocr_result/COCO_train2014_000000268711.jpg" width=425 height=282/>

<img src="/ocr_result/20180415_164208.jpg" width=360 height=640/><img src="/ocr_result/20180416_173028.jpg" width=360 height=640/>
