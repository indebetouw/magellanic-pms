
import pandas as pd

def load_phot(region='n159-all',fuse='vis'):
    savephotdir2 = '/Users/toneill/N159/photometry/FINAL_PHOT/'
    photdf = pd.read_csv(savephotdir2+f'{region}/{region}_phot_{fuse}.csv')
    return photdf

if __name__ == '__main__':
    savephotdir2 = '/Users/toneill/N159/photometry/FINAL_PHOT/'

    fuses = ['vis','ir','vis.ir']
    regions = ['e','w','s','all']
    for fuse in fuses:

        edf = pd.read_csv(savephotdir2+'n159-e/'+f'n159-e_phot_{fuse}.csv')
        wdf = pd.read_csv(savephotdir2+'n159-w/'+f'n159-w_phot_{fuse}.csv')
        sdf = pd.read_csv(savephotdir2+'n159-s/'+f'n159-s_phot_{fuse}.csv')
        alldf =pd.read_csv(savephotdir2+'n159-all/'+f'n159-all_phot_{fuse}.csv')
        offdf = pd.read_csv(savephotdir2+'off-point/'+f'off-point_phot_{fuse}.csv')

        print(f'\n {fuse}')
        print(f'E: {len(edf)}')
        print(f'W: {len(wdf)}')
        print(f'S: {len(sdf)}')
        print(f'All: {len(alldf)}')#'''
        print(f'Off: {len(offdf)}')  # '''
