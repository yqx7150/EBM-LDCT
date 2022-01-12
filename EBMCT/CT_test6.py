from models import ResNet32Large,ResNet128
import numpy as np
import os.path as osp
from tensorflow.python.platform import flags
import tensorflow as tf
import imageio
import scipy.io as io
import cv2
import matplotlib.pyplot as plt
import odl
import dicom
import glob
import time
from utils import optimistic_restore
from skimage.measure import compare_psnr,compare_ssim,compare_mse
111
flags.DEFINE_string('logdir', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_integer('num_steps', 200, 'num of steps for conditional imagenet sampling')
flags.DEFINE_float('step_lr', 600., 'step size for Langevin dynamics')
flags.DEFINE_integer('batch_size', 1, 'number of steps to run')#16
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_integer('resume_iter', -1, 'iteration to resume training from')
flags.DEFINE_bool('spec_norm', True, 'whether to use spectral normalization in weights in a model')
flags.DEFINE_bool('cclass', True, 'conditional models')
flags.DEFINE_bool('use_attention', False, 'using attention')
flags.DEFINE_string('result', 'CT', 'using attention')
FLAGS = flags.FLAGS
def show(image):
    plt.imshow(image,cmap='gray')
    plt.xticks([])
    plt.yticks([])
def write_images(x,image_save_path):
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(image_save_path, x)
def rescale_im(im):
    return np.clip(im * 256, 0, 255).astype(np.uint8)
if __name__ == "__main__":
    #========================================================================================

    model = ResNet128(num_filters=64)

    #model = ResNet32Large(num_filters=128)
    X_NOISE = tf.placeholder(shape=(None, 512, 512, 1), dtype=tf.float32)
    LABEL = tf.placeholder(shape=(None, 1), dtype=tf.float32)
    sess = tf.InteractiveSession()
    # Langevin dynamics algorithm
    weights = model.construct_weights("context_0")
    x_mod = X_NOISE
    x_mod = x_mod + tf.random_normal(tf.shape(x_mod),mean=0.0,stddev=0.0001)
    energy_noise = energy_start = model.forward(x_mod, weights, label=LABEL, reuse=True, stop_at_grad=False, stop_batch=True)
    x_grad = tf.gradients(energy_noise, [x_mod])[0]
    energy_noise_old = energy_noise

    lr = FLAGS.step_lr
    x_last = x_mod - (lr) * x_grad
    x_mod = x_last
    x_mod = tf.clip_by_value(x_mod, 0, 1)
    x_output  = x_mod

    sess.run(tf.global_variables_initializer())
    saver = loader = tf.train.Saver()
    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    model_file = osp.join(logdir, 'model_{}'.format(FLAGS.resume_iter))
    saver.restore(sess, model_file)


    #============================================================================================
    np.random.seed(1)
    #lx = np.random.permutation(1)[:16] #16个0~1000的随机数
    lx = np.random.randint(0, 1, (16))

    ims = []
    PSNR=[]
    #im_complex=np.zeros((16,512,512))
    im_best=np.zeros((512,512))

##############################################################################
    N = 512
    ANG = 180
    VIEW = 360
    cols = rows =512
    THETA = np.linspace(0, ANG, VIEW + 1)
    THETA = THETA[:-1]
    #A = lambda x: radon(x, THETA, circle=False).astype(np.float32)
    #AT = lambda y: iradon(y, THETA, circle=False, filter=None, output_size=N).astype(np.float32)/(np.pi/(2*len(THETA)))
    #AINV = lambda y: iradon(y, THETA, circle=False, output_size=N).astype(np.float32)

    angle_partition = odl.uniform_partition(0, 2 * np.pi, 1000)
    detector_partition = odl.uniform_partition(-360, 360, 1000)
    geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition,
	                        src_radius=500, det_radius=500)

    reco_space = odl.uniform_discr(min_pt=[-128, -128], max_pt=[128, 128], shape=[512, 512], dtype='float32')
    #src_radius = 870
    #det_radius = 1270
    #geometry = odl.tomo.cone_beam_geometry(reco_space, src_radius, det_radius,
    #                               num_angles=720)

    ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
    pseudoinverse = odl.tomo.fbp_op(ray_trafo)

    imas_path = glob.glob('./Test_6/*.IMA')
    imas_path.sort()
    imgidex = 0
    psnr_all = np.zeros(len(imas_path))
    ssim_all = np.zeros(len(imas_path))
    for jj in range(0,1):
        dataset=dicom.read_file(imas_path[jj])
        #dataset = dicom.read_file('./L067_FD_1_1.CT.0001.0001.2015.12.22.18.09.40.840353.358074219.IMA')
        img1 = dataset.pixel_array.astype(np.float32)    ## 读取像素范围 0-4096
        img = img1
        RescaleSlope = dataset.RescaleSlope
        RescaleIntercept = dataset.RescaleIntercept
        CT_img = img * RescaleSlope + RescaleIntercept
        ATA = ray_trafo.adjoint(ray_trafo(ray_trafo.domain.one()))
        ## LOW-DOSE SINOGRAM GENERATION
        photons_per_pixel = 1e4 #5e4#
        mu_water = 0.02
        #epsilon = 0.0001
        #print('aaaaaa',ray_trafo.range)
        phantom = reco_space.element(img)
        #nonlinear_operator = odl.ufunc_ops.exp(ray_trafo.range) * (- mu_water * ray_trafo)
        phantom = phantom/1000.0
        #proj_data = nonlinear_operator(phantom)
        #proj_data = odl.phantom.poisson_noise(proj_data * photons_per_pixel) / photons_per_pixel
        #proj_data = -np.log(proj_data)
        #image_input = pseudoinverse(proj_data)
        proj_data = ray_trafo(phantom)
        proj_data = np.exp(-proj_data * mu_water)
        proj_data = odl.phantom.poisson_noise(proj_data * photons_per_pixel)
        proj_data = np.maximum(proj_data, 1) / photons_per_pixel
        proj_data = np.log(proj_data) * (-1 / mu_water)
        image_input = pseudoinverse(proj_data)
        image_input = image_input
        x = np.copy(image_input)
        z = np.copy(x)
        maxdegrade = np.max(image_input)
        image_gt = (CT_img-np.min(CT_img))/(np.max(CT_img)-np.min(CT_img))
        #print(image_input.shape,np.max(image_gt),np.max(image_input))
        image_input = image_input.asarray()
        #print(np.max(image_input))
        #assert False
        image_input_show = image_input.copy()
        psnr_ori = compare_psnr(255*abs(image_input/maxdegrade),255*abs(image_gt),data_range=255)
        ssim_ori = compare_ssim(abs(image_input/maxdegrade),abs(image_gt),data_range=1)
        image_input = image_input/maxdegrade
        image_gt = (CT_img-np.min(CT_img))/(np.max(CT_img)-np.min(CT_img))
        #image_gt = image_gt/255.0
        #image_input = image_input/255.0
        image_shape = list((1,)+(1,)+image_input.shape[0:2])
    ##############################################################################


        x_mod = np.random.uniform(0, 1, size=(FLAGS.batch_size, 512, 512, 1))#[16,256,256,3]
        #labels = np.eye(1000)[lx]
        labels = np.eye(1)[lx]
        max_psnr=-100
        
        
        SSIM_ALL=[]
        for i in range(FLAGS.num_steps):
            sstart_in = time.time()
            e, im_complex,grad= sess.run([energy_noise,x_output,x_grad],{X_NOISE:x_mod, LABEL:labels})
            psnr_sum=0
            psnr_ave=0
            im_complex=np.squeeze(im_complex)
            #cv2.imwrite('./noise.png',im_complex)
	    ################################################
            #print('im_complex:',np.max(im_complex),np.min(im_complex))
            #print(im_complex.shape)
            psnr1 = compare_psnr(255*abs(im_complex/np.max(im_complex)),255*abs(image_gt),data_range=255)
            ssim1 = compare_ssim(abs(im_complex),abs(image_gt),data_range=1)  
            '''
            fig = plt.figure(1) 
            fig.suptitle("(%d)/2000"%(i+1))      
            plt.subplot(221)
            plt.imshow(CT_img, cmap='gray', vmin=-1150, vmax=350)
            plt.axis('off')
            plt.axis('image')
            plt.title('full dose')
            #############################################
            
            #######################
            plt.subplot(222)
            plt.imshow(image_input_show*1000-1024, cmap='gray', vmin=-1150, vmax=350)
            plt.axis('off')
            plt.axis('image')
            plt.title('quarater dose:%.3f/%.4f'%(psnr_ori,ssim_ori))
            ######################
            
            #### $$$$$$$$$$$$$ imshow $$$$$$$$$$$$$ #####
            plt.subplot(223)
            plt.imshow(im_complex*maxdegrade*1000-1000, cmap='gray', vmin=-1150, vmax=350)
            plt.axis('off')
            plt.axis('image')
            plt.title('glow_gen: %.3f/%.4f'%(psnr1,ssim1))
            '''
            ##############################SQS################################
            hyper=60
            sum_diff =  x - im_complex*maxdegrade #先验项差值
            norm_diff = ray_trafo.adjoint((ray_trafo(x) - proj_data)) #保真项差值
            x_new = z - (norm_diff + 2*hyper*sum_diff)/(ATA + 2*hyper) #更新
            z = x_new + 0.5 * (x_new - x)
            x = x_new
            x_rec = x.asarray()
            x_rec = x_rec/maxdegrade        
            #################################################################
            #print(x_rec.shape)
            psnr=compare_psnr(255*abs(x_rec/np.max(x_rec)),255*abs(image_gt),data_range=255)
            ssim = compare_ssim(abs(x_rec),abs(image_gt),data_range=1)  
            SSIM_ALL.append(ssim)
            '''
            plt.subplot(224)
            plt.imshow(maxdegrade*x_rec*1000-1000, cmap='gray', vmin=-1150, vmax=350)
            plt.axis('off')
            plt.axis('image')
            plt.title('data fidelity: %.3f/%.4f'%(psnr,ssim))
            fig.savefig('./result/test_ct/{}/iter_{}_.png'.format(jj,i))
            #fig.savefig('./result/{}/iter_{}_.png'.format(FLAGS.result,i))
            plt.pause(0.1)
            '''

            #cv2.imwrite('./noise.png',im_complex[:,:,0]*255)
            #cv2.imwrite('./gengxin.png',maxdegrade*x_rec*1000-1000)
            #cv2.imwrite('./truth.png',CT_img)
            
            '''
            psnr=compare_psnr(255*abs(x_rec),255*abs(image_gt),data_range=255)
            PSNR.append(psnr)
            psnr_sum+=psnr
            
            psnr_ave=(psnr_sum/FLAGS.batch_size)
            psnr=max(PSNR)
            id=PSNR.index(max(PSNR))
            print("step:{}".format(i),' id:',id,' PSNR:', psnr,' PSNR_ave:',psnr_ave)
            PSNR=[]
            '''
            #############################best

            im_best=x_rec
            ##############################
            x_mod[:,:,:,0]=x_rec
            print('jj:',jj,'iter:',i,'max_psnr:',psnr)
            
            if psnr>max_psnr:
                max_psnr=psnr
                '''
                ct_gruth=np.clip(CT_img,-1150,350)
                ct_gruth = (ct_gruth-np.min(ct_gruth))/(np.max(ct_gruth)-np.min(ct_gruth))
                cv2.imwrite('./result/test_6_1e5/{}/gt.png'.format(jj),255*ct_gruth)
                ct_ld=image_input_show*1000-1024
                ct_ld=np.clip(ct_ld,-1150,350)
                ct_ld = (ct_ld-np.min(ct_ld))/(np.max(ct_ld)-np.min(ct_ld))  
                cv2.imwrite('./result/test_6_1e5/{}/ld_{}_{}.png'.format(jj,psnr_ori,ssim_ori),255*ct_ld) 
                
                ct_rec=maxdegrade*x_rec*1000-1000
                ct_rec=np.clip(ct_rec,-1150,350)
                ct_rec = (ct_rec-np.min(ct_rec))/(np.max(ct_rec)-np.min(ct_rec))  
                cv2.imwrite('./result/test_6_1e5/{}/rec_{}_{}.png'.format(jj,max_psnr,ssim),255*ct_rec) 
                io.savemat('./result/test_6_1e5/{}/rec.mat'.format(jj),{'img':maxdegrade*x_rec*1000})
            print('jj:',jj,'iter:',i,'max_psnr:',max_psnr) 
        
            	'''
        #io.savemat('./result/test_6_1e5/{}/ssim.mat'.format(jj),{'result':SSIM_ALL})
        #psnr=compare_psnr(255*abs(im_complex[0]),255*abs(ori_Image),data_range=255)
        #ssim=compare_ssim(abs(im_complex[0]),abs(ori_Image),data_range=1)
        #rmse=compare_mse(abs(im_complex[0]),abs(ori_Image))
        #print("step:{} ".format(i),'PSNR :', psnr,'SSIM :', ssim,'RMSE :',rmse)

        #plt.ion()
        #show(abs(im_best))
        #plt.pause(0.3)
        #im_rec=((x_mod[:,:,:,0]+x_mod[:,:,:,2])/2)+1j*x_mod[:,:,:,1] 
        #write_images(abs(im_rec[id]),osp.join('./result/'+'CT_recon_max'+'.png'))
        #ims.append(rescale_im(abs(im_rec)).reshape((1, 1, 512, 512)).transpose((0, 2, 1, 3)).reshape((512, 512)))
    #imageio.mimwrite('MRI-sample_max.gif', ims)
