from pathlib import Path
from sdt import  roi, io#, loc, motion, multicolor, nbui, brightness,

import subprocess
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import itertools
from scipy.stats import norm, gaussian_kde
import seaborn as sns


'------------------------------------------------------------------------------------------------------------------------------------'

def create_roi(top_left, bottom_right, data_sm, data_files_sm_filename, data_brightness, data_files_brightness_filename):
#create roi based on image-j coordinates    
    roi_imagej = roi.ROI((top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]))

#size of the chosen roi in um, based on the coordinates from imagej
    size=[bottom_right[0]-top_left[0],bottom_right[1]-top_left[1]]

#size of the chosen roi in um, based on the coordinates from imagej/image size
    size_um=[size[0]*0.160, size[1]*0.160]
    
#print('ROI size in pixels:  '+ str(size)+'\n' +'ROI size in um:  '+ str(size_um))

    #cut & save roi of the data
    #data_imgs, data_roi=[], []
    
    for file in range(0,len(data_files_sm_filename)):
        data_files_sm = sorted(Path(data_sm).glob(data_files_sm_filename[file]+"*.SPE")) #.tiff
        data_imgs=[io.ImageSequence(b).open() for b in data_files_sm]

    #assign left and right roi to all stacks
        data_roi=[roi_imagej(i) for i in data_imgs]

   # for file_n in data_files_sm_filename:
        for i in range(0, len(data_files_sm)):
            io.save_as_tiff(f'{data_sm}/'+data_files_sm_filename[file]+'_roi_'+str(i)+'.tiff', data_roi[i])

    for file_b in range(0,len(data_files_brightness_filename)):
        data_files_brightness = sorted(Path(data_brightness).glob(data_files_brightness_filename[file_b]+"*.SPE")) #.tiff
        data_imgs_b=[io.ImageSequence(b).open() for b in data_files_brightness]

    #assign left and right roi to all stacks
        data_roi_b=[roi_imagej(i) for i in data_imgs_b]

   # for file_n in data_files_sm_filename:
        for i in range(0, len(data_files_brightness)):
            io.save_as_tiff(f'{data_brightness}/'+data_files_brightness_filename[file_b]+'_roi_'+str(i)+'.tiff', data_roi_b[i])
    
    print(f'ROI size in pixels:  {size}, ROI size in um:  {size_um}')

    return size, size_um, roi_imagej#, data_files_sm


'------------------------------------------------------------------------------------------------------------------------------------'

def sm_density(data_sm, data_files_sm_filename, sm_start_frame, size_um, dol):
    surface_density_sm, sem_sm, data_loc_sm=[], [], []
    single_molecule_mean, single_molecule_bg, single_molecule_size=[],[],[]

    for file in range(0,len(data_files_sm_filename)):

        #read and load the .h5 file
        data=sorted(Path(data_sm).glob(data_files_sm_filename[file]+'*.h5'))
        data_load = [io.load(f) for f in data]

        data_loc_sm_file, localization=[], []
        for i in range(0, len(data_load)):
            data_mask=data_load[i]['frame']==sm_start_frame
            
            #data_localization=data_loc_sm_file[k]['frame'][data_loc_mask]
            #data_loc_sm_file.append(data_load[i][data_load[i]['frame'].between(sm_start_frame, sm_final_frame, inclusive='both')])

            data_loc_sm_file.append(data_load[i]['frame'][data_mask])
        
            localization.append(len(data_loc_sm_file[i]))
            
            
        data_loc_sm.append(data_loc_sm_file)

        surface_density_sm_file=np.round(((np.nanmean(localization)/(size_um[0]*size_um[1])))/dol,5)
        surface_density_sm.append(surface_density_sm_file)

        #final plot
        surface_density_sm_list=[]

        for loc in localization:
            surface_density_sm_list.append((loc/(size_um[0]*size_um[1])/dol))

        sem_sm.append(np.round(np.std(surface_density_sm_list)/np.sqrt(np.size(surface_density_sm_list)),5))


        fig, ax = plt.subplots()

        ax.plot(surface_density_sm_list, 'o')

        ax.set_xlabel('positions')
        ax.set_ylabel(r'surface density [molecules/$\mu m^2$]')
        ax.set_title(f'Sample: {data_files_sm_filename[file]} \n Average surface density (dol = {dol}): {surface_density_sm[file]:.3f}'+r'$\pm$'+f'{sem_sm[file]:.3f} molecules/'+r'$\mu m^2$')
        fig.savefig(f'{data_sm}/'+data_files_sm_filename[file]+"_density.png",bbox_inches='tight')


        print(f'surface density of single-molecule data (dol= {dol}) {data_files_sm_filename[file]}: {surface_density_sm[file]} molecules/um^2')

    
    return surface_density_sm, sem_sm


'------------------------------------------------------------------------------------------------------------------------------------'
def brightness(data_brightness, data_files_brightness_filename, brightness_start_frame, brightness_final_frame, size_um, dol):
    surface_density_sm, sem_sm, data_loc_brightness=[], [], []
    single_molecule_mean, single_molecule_bg, single_molecule_size=[],[],[]

    for file in range(0,len(data_files_brightness_filename)):

        #read and load the .h5 file
        data=sorted(Path(data_brightness).glob(data_files_brightness_filename[file]+'*.h5'))
        data_load = [io.load(f) for f in data]

        data_loc_brightness_file=[]
        for i in range(0, len(data_load)):
            data_loc_brightness_file.append(data_load[i][data_load[i]['frame'].between(brightness_start_frame, brightness_final_frame, inclusive='both')])
            
            
        data_loc_brightness.append(data_loc_brightness_file)

        
    
    
    #sm brightness: uplod brightness and background column from the localization table, make an average
        single_molecule_mean_file=[]
        single_molecule_bg_file=[]
        single_molecule_size_file=[]
        
        for i in range(0,len(data_loc_brightness[file])):

    #brightness data
            data_signal=data_loc_brightness[file][i]['mass']
            single_molecule_mean_file.append(np.nanmean(data_signal))
    
    #background data
            data_bg=data_loc_brightness[file][i]['bg']
            single_molecule_bg_file.append(np.nanmean(data_bg))

    #size data
            data_size=data_loc_brightness[file][i]['size']
            single_molecule_size_file.append(np.nanmean(data_size))

        single_molecule_mean.append(np.nanmean(single_molecule_mean_file))
        single_molecule_bg.append(np.nanmean(single_molecule_bg_file))
        single_molecule_size.append(np.nanmean(single_molecule_size_file))
        
        
        

    
    return single_molecule_mean, single_molecule_bg, single_molecule_size, data_loc_brightness


'------------------------------------------------------------------------------------------------------------------------------------'


def bulk_density(single_molecule_mean, single_molecule_bg, data_bulk, data_bulk_filename, bulk_start_frame, roi_imagej, size_um, dol, brightness_correlation=False):      
        #bulk data
    surface_density_bulk_avg, sem_bulk_avg, surface_density_bulk_ind, sem_bulk_ind, mean_bulk=[],[],[],[],[]
    
    for file_bulk in range(0,len(data_bulk_filename)):
        data_files_bulk = sorted(Path(data_bulk).glob(data_bulk_filename[file_bulk]+"*.SPE")) #.tiff
        data_imgs_bulk=[]
    
    

        for b in range(0,len(data_files_bulk)):
            with io.ImageSequence(data_files_bulk[b]) as ims:
                img = ims[bulk_start_frame]
                data_imgs_bulk.append(img)

    #assign roi to all images, calculate overall intensity over roi
        bulk_roi = [np.sum(roi_imagej(j)) for j in data_imgs_bulk]

    #calculate mean from all overall intensities over rois
        mean_int_bulk=np.nanmean(bulk_roi)
        mean_bulk.append(mean_int_bulk)

          

            #surface density: sm data are averaged
        surface_density_bulk_file_avg=np.round((((mean_int_bulk/(size_um[0]*size_um[1]))-(np.mean(single_molecule_bg)/(0.160*0.160)))/(np.mean(single_molecule_mean)))/dol,2)     
        surface_density_bulk_avg.append(surface_density_bulk_file_avg)

            #surface density:  first sm filename is used to analyse first bulk filename etc.   
        surface_density_bulk_file_ind=np.round((((mean_int_bulk/(size_um[0]*size_um[1]))-(np.mean(single_molecule_bg[file_bulk])/(0.160*0.160)))/(np.mean(single_molecule_mean[file_bulk])))/dol,2)     
        surface_density_bulk_ind.append(surface_density_bulk_file_ind)

    #final plot
        bulk_roi_list_avg,bulk_roi_list_ind =[],[]
    
        for k in bulk_roi:
            bulk_roi_list_avg.append((((k/(size_um[0]*size_um[1]))-(np.mean(single_molecule_bg)/(0.160*0.160)))/(np.mean(single_molecule_mean)))/dol)
            bulk_roi_list_ind.append((((k/(size_um[0]*size_um[1]))-(np.mean(single_molecule_bg[file_bulk])/(0.160*0.160)))/(np.mean(single_molecule_mean[file_bulk])))/dol)
    
           #sem_bulk_file=np.round(np.std(bulk_roi_list)/np.sqrt(np.size(bulk_roi_list)),2)
        sem_bulk_avg.append(np.round(np.std(bulk_roi_list_avg)/np.sqrt(np.size(bulk_roi_list_avg)),2))
        sem_bulk_ind.append(np.round(np.std(bulk_roi_list_ind)/np.sqrt(np.size(bulk_roi_list_ind)),2))



        if brightness_correlation==True:
            fig, ax = plt.subplots()
            ax.plot(bulk_roi_list_avg, 'o')
            ax.set_xlabel('positions')
            ax.set_ylabel(r'surface density [molecules/$\mu m^2$]')

                
            ax.set_title(f'Sample: {data_bulk_filename[file_bulk]} \n Average surface density (dol = {dol}): {surface_density_bulk_avg[file_bulk]:.1f}'+r'$\pm$'+f'{sem_bulk_avg[file_bulk]:.1f} molecules/'+r'$\mu m^2$')
            fig.savefig(f'{data_bulk}/'+data_bulk_filename[file_bulk]+"_density_sm_avg.png",bbox_inches='tight')
            
            print(f'surface density of bulk data (dol= {dol}) {data_bulk_filename[file_bulk]}: {surface_density_bulk_avg[file_bulk]} molecules/um^2, sm_data correlated, overall mean of sm_data used')

        else:
            fig, ax = plt.subplots()
            ax.plot(bulk_roi_list_ind, 'o')
            ax.set_xlabel('positions')
            ax.set_ylabel(r'surface density [molecules/$\mu m^2$]')
            ax.set_title(f'Sample: {data_bulk_filename[file_bulk]} \n Average surface density (dol = {dol}): {surface_density_bulk_ind[file_bulk]:.1f}'+r'$\pm$'+f'{sem_bulk_ind[file_bulk]:.1f} molecules/'+r'$\mu m^2$')
            fig.savefig(f'{data_bulk}/'+data_bulk_filename[file_bulk]+"_density_sm_ind.png",bbox_inches='tight')
            
            print(f'surface density of bulk data (dol= {dol}) {data_bulk_filename[file_bulk]}: {surface_density_bulk_ind[file_bulk]} molecules/um^2, sm_data uncorrelated, corresponding sm_data used')
                
           

    return surface_density_bulk_avg, surface_density_bulk_ind, mean_bulk



'------------------------------------------------------------------------------------------------------------------------------------'
def smdensity_brightness(data_sm, surface_density_sm, sem_sm):
    fig,ax=plt.subplots()

    for i in range(0,len(surface_density_sm)):
        ax.errorbar(x=i, y=surface_density_sm[i], yerr=sem_sm[i], capsize=3, color='green')
    
    ax.plot(surface_density_sm, '-o', c='green', label='surface_density_sm')
    ax.set_xlabel('sm filenames')
    
    ax.set_ylabel(r'surface density [molecules/$\mu m^2$]')
    
    ax.legend() 


    fig.savefig(f'{data_sm}/'+"_sm_density_overall.png",bbox_inches='tight')
    plt.close(fig)
    
    return fig

'---------------------------------------------------------------------------------------------------------------------------------------'
def bulkdensity_brightness(data_bulk, surface_density_bulk_avg,surface_density_bulk_ind, mean_bulk, surface_density_sm, single_molecule_mean):
    fig,ax=plt.subplots(1,3, figsize=(12, 4))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    #sm brightness data
    ax[0].plot(single_molecule_mean, '-o', c='green', label='mean sm brightness')
    ax[0].set_xlabel('sm filenames')    
    ax[0].set_ylabel('brightness [au]')
    ax[0].legend()

    #bulk data: from individual/mean sm brightness
    ax[1].plot(surface_density_bulk_avg, '-o', label='surface_density_bulk_avg')
    ax[1].plot(surface_density_bulk_ind, '-o', label='surface_density_bulk_ind')
    ax[1].set_xlabel('bulk filenames')
    ax[1].set_ylabel(r'surface density [molecules/$\mu m^2$]')
    ax[1].legend()

    #bulk data: mean bulk intensity
    ax[2].plot(mean_bulk, '-o', c='black', label='mean bulk intensity')
    ax[2].set_xlabel('bulk filenames')
    ax[2].set_ylabel('intensity [au]')
    ax[2].legend()

    

    fig.savefig(f'{data_bulk}/'+"_bulk_density_overall.png",bbox_inches='tight')
    plt.close(fig)
    
    return fig



'------------------------------------------------------------------------------------------------------------------------------------'
def pdf_plot(data_sm, data_loc_sm, single_molecule_mean):
    cols=sns.color_palette("gnuplot", n_colors=len(data_loc_sm)) #ns.color_palette("Spectral", as_cmap=True)
  #  cols_g=sns.color_palette("Greys", n_colors=len(data_loc_sm))
    
    fig,ax=plt.subplots()
    
    overall, bins, param, max_index, x_at_max_pdf=[],[],[], [], []
    for j in range(0,len(data_loc_sm)):
        
        data_signal=[]
        for i in range(0,len(data_loc_sm[j])):

    #brightness data
            data_signal.append(data_loc_sm[j][i]['mass'])
            #single_molecule_mean_file.append(np.mean(data_signal))
#data_loc_sm[0]['mass']

        overall.append(list(itertools.chain.from_iterable(data_signal)))
        bin=np.bincount(overall[j])
        bins.append(np.arange(np.min(overall[j]),np.max(overall[j]),100))


       # gaussian_kde(dataset)
        param.append(norm.fit(overall[j]))

    #get the maximum y-value (for the pdf)
        max_index.append(np.argmax(norm.pdf(bins[j], param[j][0],  param[j][1])))
    # Get corresponding x value
        x_at_max_pdf.append(bins[j][max_index[j]])


        
   #     ax.hist(overall[j], bins=bins[j], density=True,  color=cols[j], alpha=0.4 , label=f'sm filename {j}, mean brightness= {int(single_molecule_mean[j])}')
       # ax.plot(bins[j], norm.pdf(bins[j], *param[j]), '--' , color=cols[j], label=f'PDF (Gauss) filename {j}, maximum brightness= {int(x_at_max_pdf[j])}')#+ str(norm.pdf(param[0], param[0], param[1])))
        lim=20000
        x_kde=np.linspace(0, lim, int(lim/4))
        y_kde= gaussian_kde(overall[j]).pdf(x_kde)
        ax.plot(x_kde,y_kde, color=cols[j], label=f'PDF (Gauss_kde) filename {j}, maximum brightness= {int(x_at_max_pdf[j])}')


    ax.set_xlabel('brightness')
    ax.set_ylabel('PDF')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #ax.set_xscale('log')
    fig.savefig(f'{data_sm}/'+"_brightness.png",bbox_inches='tight')
    plt.close(fig)

    ax.set_xlim((0,10000))

    return fig

'-----------------------------------------------------------------------------------------------------------------------------------'
def size_plot(data_sm, data_loc_sm, single_molecule_size):
    cols=sns.color_palette("husl", n_colors=len(data_loc_sm)) #ns.color_palette("Spectral", as_cmap=True)

    fig,ax=plt.subplots()
    
    overall, bins, param, max_index, x_at_max_pdf=[],[],[], [], []
    for j in range(0,len(data_loc_sm)):
        
        data_signal=[]
        for i in range(0,len(data_loc_sm[j])):

    #brightness data
            data_signal.append(data_loc_sm[j][i]['size'])
            #single_molecule_mean_file.append(np.mean(data_signal))
#data_loc_sm[0]['mass']

        overall.append(list(itertools.chain.from_iterable(data_signal)))
        bin=np.bincount(overall[j])
        bins.append(np.linspace(np.min(overall[j]),np.max(overall[j]),100))



        param.append(norm.fit(overall[j]))

    #get the maximum y-value (for the pdf)
        max_index.append(np.argmax(norm.pdf(bins[j], param[j][0],  param[j][1])))
    # Get corresponding x value
        x_at_max_pdf.append(bins[j][max_index[j]])


        
        ax.hist(overall[j], bins=bins[j], density=True,  color=cols[j], alpha=0.5 , label=f'sm filename {j}, mean size= {single_molecule_size[j]:.2f}')
        ax.plot(bins[j], norm.pdf(bins[j], *param[j]),  color=cols[j], label=f'PDF filename {j}, maximum size= {x_at_max_pdf[j]:.2f}')#+ str(norm.pdf(param[0], param[0], param[1])))


    ax.set_xlabel('size [px]')
    ax.set_ylabel('PDF')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig(f'{data_sm}/'+"_brightness.png",bbox_inches='tight')
    plt.close(fig)

    return fig

'------------------------------------------------------------------------------------------------------------------------------------------'
def size_brightness_plot(data_sm, data_loc_sm):
    cols=sns.color_palette("husl", n_colors=len(data_loc_sm)) #ns.color_palette("Spectral", as_cmap=True)

    fig,ax=plt.subplots()
    
    #data_size, data_mass=[],[]
    #overall, bins, param, max_index, x_at_max_pdf=[],[],[], [], []
    for j in range(0,len(data_loc_sm)):
        
        #data_size_file, data_mass_file=[],[]
        for i in range(0,len(data_loc_sm[j])):

    
            ax.plot(data_loc_sm[j][i]['size'], data_loc_sm[j][i]['mass'], '.', color=cols[j], alpha=0.1)
          
 
    ax.set_xlabel('size [px]')
    ax.set_ylabel('brightness')
    ax.set_yscale('log')
   # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig(f'{data_sm}/'+"_brightness_size.png",bbox_inches='tight')
    plt.close(fig)

    return fig
'--------------------------------------------------------------------------------------------------------------------'
def frame_brightness(data_loc_sm,sm_start_frame, sm_final_frame):

    frame_file,frame_all_file=[],[]

    fig,ax=plt.subplots()
    for file in range(0, len(data_loc_sm)):
        data_signal, frame_overall, frame_all, frame_overall=[], [], [],[]
  #  cols=sns.color_palette("viridis", n_colors=len(data_loc_sm[0]))
        for i in range(0,len(data_loc_sm[file])):
            frame_list=[]
            for frame in range(sm_start_frame, sm_final_frame+1):
                frame_list.append(np.nanmean(data_loc_sm[file][i][data_loc_sm[file][i]['frame']==frame]['mass']))
       # frame_list['mass']

    
            frame_overall.append(frame_list)

        frame_file.append(frame_overall)
        #ax.plot(frame_overall[i], color=cols[i], alpha=0.5)



        frame_all=[]

        for k in range(sm_start_frame, sm_final_frame+1):
            frame0=[]
            for i in range(0, len(frame_file[file])):
                frame0.append((frame_file[file][i][k]))
    #print(len(frame0))
            frame_all.append(np.nanmean(frame0))

        frame_all_file.append(frame_all)
    #mean.append(np.mean(frame_overall[i]))

    
        ax.plot(frame_all_file[file], label=f'mean brightness, sm filename {file}')

    
    ax.set_xlabel('frame')
    ax.set_ylabel('brightness')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig(f'{data_sm}/'+"_brightness_frame.png",bbox_inches='tight')
    plt.close(fig)

    return fig

'--------------------------------------------------------------------------------------------------------------------'

def brightness_final(data_sm, data_loc_sm, single_molecule_size, single_molecule_mean,
                     sm_start_frame, sm_final_frame):
    import numpy as np
    import itertools
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    # Plausibilität
    if len(data_loc_sm) == 0:
        raise ValueError("data_loc_sm ist leer")
    if sm_final_frame < sm_start_frame:
        raise ValueError("sm_final_frame < sm_start_frame")

    cols = sns.color_palette("pastel", n_colors=len(data_loc_sm))

    fig, ax = plt.subplots(1, 4, figsize=(20, 6))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    # 1) KDE der brightness
    for j in range(len(data_loc_sm)):
        data_signal = [data_loc_sm[j][i]['mass'] for i in range(len(data_loc_sm[j]))]
        x = np.asarray(list(itertools.chain.from_iterable(data_signal)), dtype=float)
        x = x[np.isfinite(x)]
        if x.size >= 2 and np.nanstd(x) > 0:
            x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
            if np.isfinite(x_min) and np.isfinite(x_max) and x_min < x_max:
                xv = np.linspace(max(0.0, x_min), x_max, 256)
                kde = gaussian_kde(x)
                ax[0].plot(xv, kde.pdf(xv), color=cols[j])
    ax[0].set_xlabel('brightness')
    ax[0].set_ylabel('PDF')
    ax[0].set_xlim((0, 10000))

    # 2) KDE der size
    for j in range(len(data_loc_sm)):
        data_size = [data_loc_sm[j][i]['size'] for i in range(len(data_loc_sm[j]))]
        s = np.asarray(list(itertools.chain.from_iterable(data_size)), dtype=float)
        s = s[np.isfinite(s)]
        if s.size >= 2 and np.nanstd(s) > 0:
            s_min, s_max = float(np.nanmin(s)), float(np.nanmax(s))
            if np.isfinite(s_min) and np.isfinite(s_max) and s_min < s_max:
                sv = np.linspace(max(0.0, s_min), s_max, 256)
                kde = gaussian_kde(s)
                ax[1].plot(sv, kde.pdf(sv), color=cols[j])
    ax[1].set_xlabel('size [px]')
    ax[1].set_ylabel('PDF')

    # 3) Scatter brightness vs size (nur positive brightness für log-Achse)
    for j in range(len(data_loc_sm)):
        for i in range(len(data_loc_sm[j])):
            mass = np.asarray(data_loc_sm[j][i]['mass'], dtype=float)
            size = np.asarray(data_loc_sm[j][i]['size'], dtype=float)
            mask = np.isfinite(mass) & np.isfinite(size) & (mass > 0)
            if np.any(mask):
                ax[2].plot(mass[mask], size[mask], '.', color=cols[j], alpha=0.2)
    ax[2].set_xlabel('brightness')
    ax[2].set_ylabel('size [px]')
    ax[2].set_xscale('log')

    # 4) Zeitverlauf der mittleren brightness über Frames (robust, ohne absolute Indizes)
    frame_all_file = []
    for fidx in range(len(data_loc_sm)):
        series = []
        for fr in range(sm_start_frame, sm_final_frame + 1):
            vals = []
            for i in range(len(data_loc_sm[fidx])):
                df = data_loc_sm[fidx][i]
                if 'frame' not in df or 'mass' not in df:
                    continue
                sel = df['frame'] == fr
                if np.any(sel):
                    vals.append(np.nanmean(df.loc[sel, 'mass']))
            series.append(np.nan if len(vals) == 0 else np.nanmean(vals))
        frame_all_file.append(series)
        ax[3].plot(series, color=cols[fidx], label=f'filename {fidx}')

    ax[3].set_xlabel('frame')
    ax[3].set_ylabel('brightness')
    ax[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.savefig(f'{data_sm}/'+"_brightness_overall.png", bbox_inches='tight')
    return fig
