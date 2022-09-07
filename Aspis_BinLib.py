import pandas as pd
import numpy as np

class BinLib:
    def __init__(self):
        self.R_Earth = 6378
        self.R_down = 6378 + 250
        self.R_up = 6378 + 400
        self.pos = 6378 + 350
        self.Elevations_bins = np.array([0, 1, 2, 3, 4])
        self.Elevations_bins_count = len(self.Elevations_bins)
        self.Azimuths_bins = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
        self.bins = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                             [0, 1, 2, 3, 4, 5, 6, 7, None, None, None, None, None, None, None, None],
                             [0, 1, 2, 3, None, None, None, None, None, None, None, None, None, None, None, None],
                             [0, 1,  None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                             [0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]])
        self.alphas = np.array([0.34906585, 
                                0.53709407, 
                                0.75909321, 
                                0.98136656, 
                                1.1679699,
                                np.pi/2])
        self.betas_down = np.array([1.1296167, 
                                    0.97337904, 
                                    0.77273615, 
                                    0.56441273, 
                                    0.38680733])
        self.betas_up = np.array([1.0848581, 
                                  0.94158845, 
                                  0.75137989, 
                                  0.55046280,  
                                  0.37780937])
        self.gammas_down = np.array([0.092113805, 
                                     0.060323220, 
                                     0.038966961, 
                                     0.025017033, 
                                     0.016019067])
        self.gammas_up = np.array([0.13687234, 
                                   0.092113805, 
                                   0.060323220, 
                                   0.038966961, 
                                   0.025017033])
        self.xs_down = np.array([648.79437, 
                                 465.06084, 
                                 355.92246, 
                                 298.25408, 
                                 270.82755])
        self.xs_up = np.array([984.18007, 
                               725.63486, 
                               563.26003, 
                               475.00873, 
                               432.49765])
        self.xs = np.array([335.38570, 
                            260.57401, 
                            207.33756, 
                            176.75464, 
                            161.67010])

        
    def calc_gnd_geo(self, alpha, Azimuth, r, gnd_lat, gnd_lon, input_unit_x="Deg", input_unit_gnd="Deg", output_unit_x="Deg"):
        if input_unit_x=="Deg": 
            alpha=np.radians(alpha)
            Azimuth=np.radians(Azimuth)
        if input_unit_gnd=="Deg": 
            gnd_lat=np.radians(gnd_lat)
            gnd_lon=np.radians(gnd_lon)
            
        x_size = np.sqrt(self.R_Earth*self.R_Earth+ r*r  + 2*self.R_Earth*r*np.sin(alpha))
        x_lat = np.arcsin(((self.R_Earth + r*np.sin(alpha))*np.sin(gnd_lat) + r*np.cos(alpha)*np.cos(Azimuth)*np.cos(gnd_lat))/x_size)
        citatel = (self.R_Earth+r*np.sin(alpha))*np.cos(gnd_lat)*np.sin(gnd_lon) - r*np.cos(alpha)*(np.cos(Azimuth)*np.sin(gnd_lat)*np.sin(gnd_lon) - np.sin(Azimuth)*np.cos(gnd_lon))
        menovatel = (self.R_Earth+r*np.sin(alpha))*np.cos(gnd_lat)*np.cos(gnd_lon) - r*np.cos(alpha)*(np.cos(Azimuth)*np.sin(gnd_lat)*np.cos(gnd_lon) + np.sin(Azimuth)*np.sin(gnd_lon))
        x_lon = np.arctan(citatel/menovatel)
        
        x_lon[menovatel<0]=x_lon[menovatel<0]+np.pi
        if output_unit_x=="Deg":
            x_lat = np.degrees(x_lat)
            x_lon = np.degrees(x_lon)
            
        return x_lat, x_lon, x_size
    
    def calc_geo_gnd(self, lon, lat, gnd_lon, gnd_lat, input_unit_x="Deg", input_unit_gnd="Deg", output_unit_x="Deg"):
        if input_unit_x=="Deg": 
            lon=np.radians(lon)
            lat=np.radians(lat)
        if input_unit_gnd=="Deg": 
            gnd_lat=np.radians(gnd_lat)
            gnd_lon=np.radians(gnd_lon)        
        
        r_sqr = (np.cos(lat)*np.cos(lon)*self.pos-np.cos(gnd_lat)*np.cos(gnd_lon)*self.R_Earth)**2 + (np.cos(lat)*np.sin(lon)*self.pos-np.cos(gnd_lat)*np.sin(gnd_lon)*self.R_Earth)**2 + (np.sin(lat)*self.pos-np.sin(gnd_lat)*self.R_Earth)**2
        sin_alpha = (self.pos*self.pos-self.R_Earth*self.R_Earth-r_sqr)/(2*self.R_Earth*np.sqrt(r_sqr))
        cos_A = (np.sin(lat)*self.pos-(self.R_Earth+np.sqrt(r_sqr)*sin_alpha)*np.sin(gnd_lat))/(np.sqrt(r_sqr)*np.cos(np.arcsin(sin_alpha))*np.cos(gnd_lat))
        
        alpha, A, r = np.arcsin(sin_alpha), np.arccos(cos_A), np.sqrt(r_sqr)
        if output_unit_x=="Deg":
            alpha = np.degrees(alpha)
            A = np.degrees(A)
        
        return alpha, A, r
    
    
    def calc_x(self, alpha, input_unit="Rad"): # ✓
        if input_unit=="Deg": alpha=np.radians(alpha)
        x = (np.sin(self.calc_gamma_up_n(alpha=alpha, input_unit="Rad", output_unit="Rad"))/np.cos(alpha))*self.R_up - (np.sin(self.calc_gamma_down_n(alpha=alpha, input_unit="Rad", output_unit="Rad"))/np.cos(alpha))*self.R_down
        x[alpha==np.pi/2] = self.R_up - self.R_down
        return x
    
    
    def calc_x_down(self, alpha, input_unit="Rad"): # ✓
        if input_unit=="Deg": alpha=np.radians(alpha)
        x = (np.sin(self.calc_gamma_down_n(alpha=alpha, input_unit="Rad", output_unit="Rad"))/np.cos(alpha))*self.R_down
        x[alpha==np.pi/2] = self.R_up - self.R_down
        return x
    
    
    def calc_x_up(self, alpha, input_unit="Rad"): # ✓
        if input_unit=="Deg": alpha=np.radians(alpha)
        x = (np.sin(self.calc_gamma_up_n(alpha=alpha, input_unit="Rad", output_unit="Rad"))/np.cos(alpha))*self.R_up
        x[alpha==np.pi/2] = self.R_up - self.R_down
        return x
    
    
    def calc_x_n(self, alpha, input_unit="Rad", output="Rel"): # ✓
        if input_unit=="Deg": alpha=np.radians(alpha)
        x_n = (np.sin(self.calc_gamma_up_n(alpha=alpha, input_unit="Rad", output_unit="Rad") - self.calc_gamma_down_n(alpha=self.calc_alpha_n(alpha=alpha, input_unit="Rad", output_unit="Rad"), input_unit="Rad", output_unit="Rad"))/np.sin(self.calc_beta_down_n(alpha=alpha, input_unit="Rad", output_unit="Rad")))*self.R_up
        x_n[alpha>=self.alphas[-2]] = self.calc_x(alpha=alpha[alpha>=self.alphas[-2]], input_unit="Rad")
        if output=="Rel": x_n = x_n/self.calc_x(alpha=self.calc_alpha_n(alpha, input_unit="Rad", output_unit="Rad"), input_unit="Rad")
        elif output=="Abs": x_n = x_n
        return x_n
    
    
    def calc_x_np(self, alpha, input_unit="Rad", output="Rel"): # ✓
        if input_unit=="Deg": alpha=np.radians(alpha)
        x_np = self.calc_x(alpha=alpha, input_unit="Rad") - (np.sin(self.calc_gamma_up_n(alpha=alpha, input_unit="Rad", output_unit="Rad") - self.calc_gamma_down_n(alpha=self.calc_alpha_n(alpha=alpha, input_unit="Rad", output_unit="Rad"), input_unit="Rad", output_unit="Rad"))/np.sin(self.calc_beta_down_n(alpha=alpha, input_unit="Rad", output_unit="Rad")))*self.R_up
        x_np[alpha>=self.alphas[-2]] = self.calc_x(alpha=alpha[alpha>=self.alphas[-2]], input_unit="Rad")
        if output=="Rel": x_np = x_np/self.calc_x(alpha=self.calc_alpha_np(alpha, input_unit="Deg", output_unit="Rad"), input_unit="Rad")
        elif output=="Abs": x_np = x_np
        return x_np
    
    
    def calc_s4(self, array): # ✓
        s4 = np.sqrt(array.std()/array.mean())
        return s4
    
    
    def calc_z4(self, array, lenght): # ✓
        z4 = self.calc_s4(array)/lenght
        return z4
    
    
    def calc_Elevation_bin_n(self, alpha, input_unit="Deg"): # ✓
        if input_unit=="Deg": alpha=np.radians(alpha)      
        Elevation_bin_n = 1 * ((alpha >= self.alphas[0]) & (alpha < self.alphas[1]))
        Elevation_bin_n = Elevation_bin_n + 2 * ((alpha >= self.alphas[1]) & (alpha < self.alphas[2]))
        Elevation_bin_n = Elevation_bin_n + 3 * ((alpha >= self.alphas[2]) & (alpha < self.alphas[3]))
        Elevation_bin_n = Elevation_bin_n + 4 * ((alpha >= self.alphas[3]) & (alpha < self.alphas[4]))
        Elevation_bin_n = Elevation_bin_n + 5 * ((alpha >= self.alphas[4]))
        return Elevation_bin_n
    
    
    def calc_Azimuth_bin_n(self, alpha, Azimuth, input_unit="Deg"): # ✓
        if input_unit=="Deg": alpha=np.radians(alpha)
        if input_unit=="Rad": Azimuth=np.degrees(Azimuth)
        Azimuth_bin = np.zeros(len(alpha))
        for i in range(1, 6):
            k = 5 - i
            for j in range(0, 2**k):
                Azimuth_bin = Azimuth_bin + (j + 1) * ((self.calc_Elevation_bin_n(alpha, input_unit="Rad") == i) & (Azimuth >= j * 360 / (2**k)) & (Azimuth < (j + 1) * 360 / (2**k)))
        Azimuth_bin = Azimuth_bin.astype(np.int8)
        return Azimuth_bin
    
    
    def calc_Elevation_bin_np(self, alpha, input_unit="Deg"): # ✓
        Elevation_bin_np = self.calc_Elevation_bin_n(alpha=alpha, input_unit=input_unit) + 1
        Elevation_bin_np[Elevation_bin_np>5] = 5
        return Elevation_bin_np
    
    
    def calc_Azimuth_bin_np(self, alpha, Azimuth, input_unit="Deg"): # ✓
        Azimuth_bin_np = self.calc_Azimuth_bin_n(alpha=alpha, Azimuth=Azimuth, input_unit=input_unit)//2
        Azimuth_bin_np[Azimuth_bin_np==0] = 1
        Azimuth_bin = Azimuth_bin_np.astype(np.int8)
        return Azimuth_bin_np
    
    
    def calc_bin_n(self, alpha, Azimuth, input_unit="Deg"): # ✓
        return self.calc_Elevation_bin_n(alpha=alpha, input_unit=input_unit), self.calc_Azimuth_bin_n(alpha=alpha, Azimuth=Azimuth, input_unit=input_unit)
        
        
    def calc_bin_np(self, alpha, Azimuth, input_unit="Deg"): # ✓
        return self.calc_Elevation_bin_np(alpha=alpha, input_unit=input_unit), self.calc_Azimuth_bin_np(alpha=alpha, Azimuth=Azimuth, input_unit=input_unit)
        
        
    def calc_alpha_n(self, alpha, input_unit="Deg", output_unit="Rad"): # ✓
        if input_unit=="Deg": alpha=np.radians(alpha)
        z = np.array([self.alphas for _ in range(len(alpha))])
        w = np.reshape(alpha, (len(alpha), 1))
        alpha_n = [self.alphas[(z<=w)[i]][-1] for i in range(len(alpha))]
        if output_unit=="Deg": alpha_n = np.degrees(alpha_n)
        return alpha_n
    
    
    def calc_alpha_np(self, alpha, input_unit="Deg", output_unit="Rad"): # ✓ 
        if input_unit=="Deg": alpha=np.radians(alpha)
        z = np.array([self.alphas for _ in range(len(alpha))])
        w = np.reshape(alpha, (len(alpha), 1))
        alpha_np = [self.alphas[(z>=w)[i]][0] for i in range(len(alpha))]
        if output_unit=="Deg": alpha_np = np.degrees(alpha_np)
        return alpha_np  
        
        
    def calc_beta_down_n(self, alpha, input_unit="Deg", output_unit="Rad"): # ✓ 
        if input_unit=="Deg": alpha=np.radians(alpha)
        beta_down_n = self.calc_beta_up_n(alpha=alpha, input_unit="Rad", output_unit="Rad") + self.calc_gamma_up_n(alpha=alpha, input_unit="Rad", output_unit="Rad") - self.calc_gamma_down_n(alpha=self.calc_alpha_n(alpha=alpha, input_unit="Rad", output_unit="Rad"), input_unit="Rad", output_unit="Rad")
        if output_unit=="Deg": beta_down_n = np.degrees(beta_down_n)
        return beta_down_n
    
    
    def calc_beta_down_np(self, alpha, input_unit="Deg", output_unit="Rad"): # ✓
        if input_unit=="Deg": alpha=np.radians(alpha)
        beta_down_np = np.arcsin((self.R_Earth/self.R_down)*np.cos(alpha))
        if output_unit=="Deg": beta_down_np = np.degrees(beta_down_np)
        return beta_down_np
    
    
    def calc_beta_up_n(self, alpha, input_unit="Deg", output_unit="Rad"): # ✓
        if input_unit=="Deg": alpha=np.radians(alpha)
        beta_up_n = np.arcsin((self.R_Earth/self.R_up)*np.cos(alpha))
        if output_unit=="Deg": beta_up_n = np.degrees(beta_up_n)
        return beta_up_n
    
    
    def calc_beta_up_np(self, alpha, input_unit="Deg", output_unit="Rad"): # ✓ 
        beta_up_np = self.calc_beta_down_n(alpha=alpha, input_unit=input_unit, output_unit=output_unit)
        return beta_up_np
    
    
    def calc_gamma_down_n(self, alpha, input_unit="Deg", output_unit="Rad"): # ✓
        if input_unit=="Deg": alpha=np.radians(alpha)
        gamma_down_n = (np.pi/2)-self.calc_beta_down_np(alpha, input_unit="Rad", output_unit="Rad")-alpha
        if output_unit=="Deg": gamma_down_n = np.degrees(gamma_down_n)
        return gamma_down_n
    
    
    def calc_gamma_down_np(self, alpha, input_unit="Deg", output_unit="Rad"): # ✓
        if input_unit=="Deg": alpha=np.radians(alpha)
        gamma_down_np = (np.pi/2)-self.calc_beta_up_n(alpha, input_unit="Rad", output_unit="Rad")-alpha
        if output_unit=="Deg": gamma_down_np = np.degrees(gamma_down_np)
        return gamma_down_np
    
    
    def calc_gamma_up_n(self, alpha, input_unit="Deg", output_unit="Rad"): # ✓
        if input_unit=="Deg": alpha=np.radians(alpha)
        gamma_up_n = (np.pi/2) - self.calc_beta_up_n(alpha=alpha, input_unit="Rad", output_unit="Rad") - alpha
        if output_unit=="Deg": gamma_up_n = np.degrees(gamma_up_n)
        return gamma_up_n
    
    
    def calc_gamma_up_np(self, alpha, input_unit="Deg", output_unit="Rad"): # ✓ 
        gamma_up_np = self.calc_gamma_up_n(alpha=alpha, input_unit=input_unit, output_unit=output_unit)
        return gamma_up_np
    
    
    def calc_arc_lenght_pre(self, lat1, lat2, lon1, lon2, input_unit="Deg", output_unit="Rad"): # ✓
        if input_unit=="Deg": 
            lat1=np.radians(lat1)
            lat2=np.radians(lat2)
            lon1=np.radians(lon1)
            lon2=np.radians(lon2)
        cit = np.sqrt((np.cos(lat2)*np.sin(lon2-lon1))**2 + (np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(lon2-lon1))**2)
        men = np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1)
        c = np.arctan(cit/men)
        c[c<0] = np.pi+c[c<0]
        if output_unit=="Deg": c = np.degrees(c)
        return c


    def calc_arc_lenght(self, lat1, lat2, lon1, lon2, input_unit="Deg", output_unit="Rad"): # ✓ 
        if input_unit=="Deg": 
            lat1=np.radians(lat1)
            lat2=np.radians(lat2)
            lon1=np.radians(lon1)
            lon2=np.radians(lon2)
        c = np.arccos(np.round(np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(lon1-lon2), 6))
        if output_unit=="Deg": c = np.degrees(c)
        return c


    def calc_sphere_der(self, value1, value2, lat1, lat2, lon1, lon2, input_unit="Deg"): # ✓ 
        if input_unit=="Deg": 
            lat1=np.radians(lat1)
            lat2=np.radians(lat2)
            lon1=np.radians(lon1)
            lon2=np.radians(lon2)
        der1 = 0
        der2 = ((value2-value1)/(lat2-lat1))/(1000*self.R_Earth)
        der3 = ((value2-value1)/(lon2-lon1))/(1000*self.R_Earth*np.sin(lat2))
        der = np.array([der1, der2, der3])
        return der


    def calc_near_bin(self, lat_bins, lon_bins, lat, lon, input_bin_unit="Deg", input_unit="Deg", output_unit="Deg", pre_mode=False): # ✓ 
        if pre_mode: 
            distances = self.calc_arc_lenght_pre(lat_bins, lat, lon_bins, lon)
        else: 
            distances = self.calc_arc_lenght(lat_bins, lat, lon_bins, lon)
        min_index = np.argmin(distances)
        return np.array([lat_bins[min_index], lon_bins[min_index]])
    

    
    def calc_close_bin_faster(self, lat_bins_nonunique, lat_bins_unique, correspond_lon_bins, lat, lon):
        closest_lat = np.argmin(np.abs(lat_bins_unique-lat))
        temp_lat_bins = np.array([lat_bins_unique[closest_lat]]*len(correspond_lon_bins[closest_lat]))
        lat_bins, lon_bins = self.calc_near_bin(temp_lat_bins, np.array(correspond_lon_bins[closest_lat]), lat, lon)
        return lat_bins, lon_bins


# if __name__ == '__main__':
#     pass