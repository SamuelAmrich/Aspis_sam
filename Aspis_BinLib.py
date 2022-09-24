import pandas as pd
import numpy as np
import plotly.express as px

class BinLib:
    def __init__(self):
            self.R_Earth = 6378
            self.R_down = 6378 + 250
            self.R_up = 6378 + 400
            self.R_sci = 6378 + 350
            self.Elevations_bins = np.array([0, 1, 2, 3, 4])
            self.Elevations_bins_count = len(self.Elevations_bins)
            self.Azimuths_bins = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
            self.bins = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                 [0, 1, 2, 3, 4, 5, 6, 7, None, None, None, None, None, None, None, None],
                                 [0, 1, 2, 3, None, None, None, None, None, None, None, None, None, None, None, None],
                                 [0, 1,  None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                                 [0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]])
            self.Elevations = np.array([0.34906585, 
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
        
        
    def obs_to_geo(self, Elevation_obs, Azimuth_obs, dis_obs, lat_obs, lon_obs, input_unit_obs="Deg", input_unit_geo="Deg", output_unit="Deg"): # ✓
        if input_unit_obs=="Deg": 
            Elevation_obs=np.radians(Elevation_obs)
            Azimuth_obs=np.radians(Azimuth_obs)
        if input_unit_geo=="Deg": 
            lat_obs=np.radians(lat_obs)
            lon_obs=np.radians(lon_obs)
           
        dis_size = np.sqrt(self.R_Earth*self.R_Earth+ dis_obs*dis_obs  + 2*self.R_Earth*dis_obs*np.sin(Elevation_obs))
        lat_geo = np.arcsin(((self.R_Earth + dis_obs*np.sin(Elevation_obs))*np.sin(lat_obs) + dis_obs*np.cos(Elevation_obs)*np.cos(Azimuth_obs)*np.cos(lat_obs))/dis_size)
        temp_citatel = (self.R_Earth+dis_obs*np.sin(Elevation_obs))*np.cos(lat_obs)*np.sin(lon_obs) - dis_obs*np.cos(Elevation_obs)*(np.cos(Azimuth_obs)*np.sin(lat_obs)*np.sin(lon_obs) - np.sin(Azimuth_obs)*np.cos(lon_obs))
        temp_menovatel = (self.R_Earth+dis_obs*np.sin(Elevation_obs))*np.cos(lat_obs)*np.cos(lon_obs) - dis_obs*np.cos(Elevation_obs)*(np.cos(Azimuth_obs)*np.sin(lat_obs)*np.cos(lon_obs) + np.sin(Azimuth_obs)*np.sin(lon_obs))
        lon_geo = np.arctan(temp_citatel/temp_menovatel)
        
        lon_geo[temp_menovatel<0]=lon_geo[temp_menovatel<0]+np.pi
        if output_unit=="Deg":
            lat_geo = np.degrees(lat_geo)
            lon_geo = np.degrees(lon_geo)
            
        return lat_geo, lon_geo, dis_size
    
    def geo_to_usr(self, lat_geo, lon_geo, lat_usr, lon_usr, input_unit_geo="Deg", input_unit_usr="Deg", output_unit="Deg"): # ✓
        if input_unit_geo=="Deg": 
            lat_geo=np.radians(lat_geo)
            lon_geo=np.radians(lon_geo)
        if input_unit_usr=="Deg": 
            lat_usr=np.radians(lat_usr)
            lon_usr=np.radians(lon_usr)        
        
        dis_sqr = (np.cos(lat_geo)*np.cos(lon_geo)*self.R_sci-np.cos(lat_usr)*np.cos(lon_usr)*self.R_Earth)**2 + (np.cos(lat_geo)*np.sin(lon_geo)*self.R_sci-np.cos(lat_usr)*np.sin(lon_usr)*self.R_Earth)**2 + (np.sin(lat_geo)*self.R_sci-np.sin(lat_usr)*self.R_Earth)**2
        sin_Elevation_usr = (self.R_sci*self.R_sci-self.R_Earth*self.R_Earth-dis_sqr)/(2*self.R_Earth*np.sqrt(dis_sqr))
        cos_Azimuth_usr = (np.sin(lat_geo)*self.R_sci-(self.R_Earth+np.sqrt(dis_sqr)*sin_Elevation_usr)*np.sin(lat_usr))/(np.sqrt(dis_sqr)*np.cos(np.arcsin(sin_Elevation_usr))*np.cos(lat_usr))
        
        Elevation_usr, Azimuth_usr, dis_usr = np.arcsin(sin_Elevation_usr), np.arccos(cos_Azimuth_usr), np.sqrt(dis_sqr)
        if output_unit=="Deg":
            Elevation_usr = np.degrees(Elevation_usr)
            Azimuth_usr = np.degrees(Azimuth_usr)
        
        return Elevation_usr, Azimuth_usr, dis_usr
    
    
    def calc_dis(self, Elevation_obs, input_unit="Rad"): # ✓  
        if input_unit=="Deg": Elevation_obs=np.radians(Elevation_obs)
        dis = (np.sin(self.calc_gamma_up_n(Elevation_obs, input_unit="Rad", output_unit="Rad"))/np.cos(Elevation_obs))*self.R_up - (np.sin(self.calc_gamma_down_n(Elevation_obs, input_unit="Rad", output_unit="Rad"))/np.cos(Elevation_obs))*self.R_down
        dis[Elevation_obs==np.pi/2] = self.R_up - self.R_down
        return dis
    
    
    def calc_dis_down(self, Elevation_obs, input_unit="Rad"): # ✓  
        if input_unit=="Deg": Elevation_obs=np.radians(Elevation_obs)
        dis_down = (np.sin(self.calc_gamma_down_n(Elevation_obs, input_unit="Rad", output_unit="Rad"))/np.cos(Elevation_obs))*self.R_down
        dis_down[Elevation_obs==np.pi/2] = self.R_up - self.R_down
        return dis_down
    
    
    def calc_dis_up(self, Elevation_obs, input_unit="Rad"): # ✓
        if input_unit=="Deg": Elevation_obs=np.radians(Elevation_obs)
        dis_up = (np.sin(self.calc_gamma_up_n(Elevation_obs, input_unit="Rad", output_unit="Rad"))/np.cos(Elevation_obs))*self.R_up
        dis_up[Elevation_obs==np.pi/2] = self.R_up - self.R_down
        return dis_up
    
    
    def calc_dis_n(self, Elevation, input_unit="Rad", output="Rel"): # ✓  
        if input_unit=="Deg": Elevation=np.radians(Elevation)
        dis_n = (np.sin(self.calc_gamma_up_n(Elevation, input_unit="Rad", output_unit="Rad") - self.calc_gamma_down_n(elf.calc_Elevation_n(Elevation, input_unit="Rad", output_unit="Rad"), input_unit="Rad", output_unit="Rad"))/np.sin(self.calc_beta_down_n(Elevation, input_unit="Rad", output_unit="Rad")))*self.R_up
        dis_n[Elevation>=self.Elevations[-2]] = self.calc_x(Elevation[Elevation>=self.Elevations[-2]], input_unit="Rad")
        if output=="Rel": dis_n = dis_n/self.calc_x(self.calc_Elevation_n(Elevation, input_unit="Rad", output_unit="Rad"), input_unit="Rad")
        elif output=="Abs": dis_n = dis_n
        return dis_n
    
    
    def calc_dis_np(self, Elevation, input_unit="Rad", output="Rel"): # ✓  
        if input_unit=="Deg": Elevation=np.radians(Elevation)
        dis_np = self.calc_x(Elevation, input_unit="Rad") - (np.sin(self.calc_gamma_up_n(Elevation, input_unit="Rad", output_unit="Rad") - self.calc_gamma_down_n(self.calc_Elevation_n(Elevation, input_unit="Rad", output_unit="Rad"), input_unit="Rad", output_unit="Rad"))/np.sin(self.calc_beta_down_n(Elevation, input_unit="Rad", output_unit="Rad")))*self.R_up
        dis_np[Elevation>=self.Elevations[-2]] = self.calc_x(Elevation[Elevation>=self.Elevations[-2]], input_unit="Rad")
        if output=="Rel": dis_np = dis_np/self.calc_x(self.calc_Elevation_np(Elevation, input_unit="Deg", output_unit="Rad"), input_unit="Rad")
        elif output=="Abs": dis_np = dis_np
        return dis_np
    
    
    def calc_s4(self, array): # ✓    
        s4 = np.sqrt(array.std()/array.mean())
        return s4
    
    
    def calc_z4(self, array, dis): # ✓    
        z4 = self.calc_s4(array)/dis
        return z4
    
    
    def calc_Elevation_bin_n(self, Elevation, input_unit="Deg"): # ✓      
        if input_unit=="Deg": Elevation=np.radians(Elevation)      
        Elevation_bin_n = 1 * ((Elevation >= self.Elevations[0]) & (Elevation < self.Elevations[1]))
        Elevation_bin_n = Elevation_bin_n + 2 * ((Elevation >= self.Elevations[1]) & (Elevation < self.Elevations[2]))
        Elevation_bin_n = Elevation_bin_n + 3 * ((Elevation >= self.Elevations[2]) & (Elevation < self.Elevations[3]))
        Elevation_bin_n = Elevation_bin_n + 4 * ((Elevation >= self.Elevations[3]) & (Elevation < self.Elevations[4]))
        Elevation_bin_n = Elevation_bin_n + 5 * ((Elevation >= self.Elevations[4]))
        return Elevation_bin_n
    
    
    def calc_Azimuth_bin_n(self, Elevation, Azimuth, input_unit="Deg"): # ✓      
        if input_unit=="Deg": Elevation=np.radians(Elevation)
        if input_unit=="Rad": Azimuth=np.degrees(Azimuth)
        Azimuth_bin = np.zeros(len(Elevation))
        for i in range(1, 6):
            k = 5 - i
            for j in range(0, 2**k):
                Azimuth_bin = Azimuth_bin + (j + 1) * ((self.calc_Elevation_bin_n(Elevation, input_unit="Rad") == i) & (Azimuth >= j * 360 / (2**k)) & (Azimuth < (j + 1) * 360 / (2**k)))
        Azimuth_bin = Azimuth_bin.astype(np.int8)
        return Azimuth_bin
    
    
    def calc_Elevation_bin_np(self, Elevation, input_unit="Deg"): # ✓      
        Elevation_bin_np = self.calc_Elevation_bin_n(Elevation=Elevation, input_unit=input_unit) + 1
        Elevation_bin_np[Elevation_bin_np>5] = 5
        return Elevation_bin_np
    
    
    def calc_Azimuth_bin_np(self, Elevation, Azimuth, input_unit="Deg"): # ✓      
        Azimuth_bin_np = self.calc_Azimuth_bin_n(Elevation=Elevation, Azimuth=Azimuth, input_unit=input_unit)//2
        Azimuth_bin_np[Azimuth_bin_np==0] = 1
        Azimuth_bin = Azimuth_bin_np.astype(np.int8)
        return Azimuth_bin_np
    
    
    def calc_bin_n(self, Elevation, Azimuth, input_unit="Deg"): # ✓      
        return self.calc_Elevation_bin_n(Elevation=Elevation, input_unit=input_unit), self.calc_Azimuth_bin_n(Elevation=Elevation, Azimuth=Azimuth, input_unit=input_unit)
        
        
    def calc_bin_np(self, Elevation, Azimuth, input_unit="Deg"): # ✓      
        return self.calc_Elevation_bin_np(Elevation=Elevation, input_unit=input_unit), self.calc_Azimuth_bin_np(Elevation=Elevation, Azimuth=Azimuth, input_unit=input_unit)
        
        
    def calc_Elevation_n(self, Elevation, input_unit="Deg", output_unit="Rad"): # ✓      
        if input_unit=="Deg": Elevation=np.radians(Elevation)
        z = np.array([self.Elevations for _ in range(len(Elevation))])
        w = np.reshape(Elevation, (len(Elevation), 1))
        Elevation_n = [self.Elevations[(z<=w)[i]][-1] for i in range(len(Elevation))]
        if output_unit=="Deg": Elevation_n = np.degrees(Elevation_n)
        return Elevation_n
    
    
    def calc_Elevation_np(self, Elevation, input_unit="Deg", output_unit="Rad"): # ✓       
        if input_unit=="Deg": Elevation=np.radians(Elevation)
        z = np.array([self.Elevations for _ in range(len(Elevation))])
        w = np.reshape(Elevation, (len(Elevation), 1))
        Elevation_np = [self.Elevations[(z>=w)[i]][0] for i in range(len(Elevation))]
        if output_unit=="Deg": Elevation_np = np.degrees(Elevation_np)
        return Elevation_np  
        
        
    def calc_beta_down_n(self, Elevation, input_unit="Deg", output_unit="Rad"): # ✓       
        if input_unit=="Deg": Elevation=np.radians(Elevation)
        beta_down_n = self.calc_beta_up_n(Elevation=Elevation, input_unit="Rad", output_unit="Rad") + self.calc_gamma_up_n(Elevation=Elevation, input_unit="Rad", output_unit="Rad") - self.calc_gamma_down_n(Elevation=self.calc_Elevation_n(Elevation=Elevation, input_unit="Rad", output_unit="Rad"), input_unit="Rad", output_unit="Rad")
        if output_unit=="Deg": beta_down_n = np.degrees(beta_down_n)
        return beta_down_n
    
    
    def calc_beta_down_np(self, Elevation, input_unit="Deg", output_unit="Rad"): # ✓      
        if input_unit=="Deg": Elevation=np.radians(Elevation)
        beta_down_np = np.arcsin((self.R_Earth/self.R_down)*np.cos(Elevation))
        if output_unit=="Deg": beta_down_np = np.degrees(beta_down_np)
        return beta_down_np
    
    
    def calc_beta_up_n(self, Elevation, input_unit="Deg", output_unit="Rad"): # ✓      
        if input_unit=="Deg": Elevation=np.radians(Elevation)
        beta_up_n = np.arcsin((self.R_Earth/self.R_up)*np.cos(Elevation))
        if output_unit=="Deg": beta_up_n = np.degrees(beta_up_n)
        return beta_up_n
    
    
    def calc_beta_up_np(self, Elevation, input_unit="Deg", output_unit="Rad"): # ✓       
        beta_up_np = self.calc_beta_down_n(Elevation=Elevation, input_unit=input_unit, output_unit=output_unit)
        return beta_up_np
    
    
    def calc_gamma_down_n(self, Elevation, input_unit="Deg", output_unit="Rad"): # ✓     
        if input_unit=="Deg": Elevation=np.radians(Elevation)
        gamma_down_n = (np.pi/2)-self.calc_beta_down_np(Elevation, input_unit="Rad", output_unit="Rad")-Elevation
        if output_unit=="Deg": gamma_down_n = np.degrees(gamma_down_n)
        return gamma_down_n
    
    
    def calc_gamma_down_np(self, Elevation, input_unit="Deg", output_unit="Rad"): # ✓      
        if input_unit=="Deg": Elevation=np.radians(Elevation)
        gamma_down_np = (np.pi/2)-self.calc_beta_up_n(Elevation, input_unit="Rad", output_unit="Rad")-Elevation
        if output_unit=="Deg": gamma_down_np = np.degrees(gamma_down_np)
        return gamma_down_np
    
    
    def calc_gamma_up_n(self, Elevation, input_unit="Deg", output_unit="Rad"): # ✓      
        if input_unit=="Deg": Elevation=np.radians(Elevation)
        gamma_up_n = (np.pi/2) - self.calc_beta_up_n(Elevation=Elevation, input_unit="Rad", output_unit="Rad") - Elevation
        if output_unit=="Deg": gamma_up_n = np.degrees(gamma_up_n)
        return gamma_up_n
    
    
    def calc_gamma_up_np(self, Elevation, input_unit="Deg", output_unit="Rad"): # ✓       
        gamma_up_np = self.calc_gamma_up_n(Elevation=Elevation, input_unit=input_unit, output_unit=output_unit)
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


    def calc_sphere_der(self, lat1, lat2, lon1, lon2, value1, value2, input_unit="Deg"): # ✓  
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


    def calc_near_bin(self, lat_bin, lon_bin, lat_geo, lon_geo, input_bin_unit="Deg", input_unit="Deg", output_unit="Deg", pre_mode=False, return_index = False): # ✓     
        if pre_mode: 
            distances = self.calc_arc_lenght_pre(lat_bin, lat_geo, lon_bin, lon_geo)
        else: 
            distances = self.calc_arc_lenght(lat_bin, lat_geo, lon_bin, lon_geo)
        min_index = np.argmin(distances)
        
        if return_index:
            min_index
        else:
            return np.array([lat_bin[min_index], lon_bin[min_index]])
    
    
    def calc_close_bin_faster(self, lat_bins_nonunique, lat_bins_unique, correspond_lon_bins, lat, lon): # ✓  
        closest_lat = np.argmin(np.abs(lat_bins_unique-lat))
        temp_lat_bins = np.array([lat_bins_unique[closest_lat]]*len(correspond_lon_bins[closest_lat]))
        lat_bins, lon_bins = self.calc_near_bin(temp_lat_bins, np.array(correspond_lon_bins[closest_lat]), lat, lon)
        return lat_bins, lon_bins
    
    
    def add_latlon(self, dataframe, number): # ✓  
        dataframe['lat'] = np.zeros(dataframe.shape[0]).astype(object)
        
        for line in range(dataframe['lat'].shape[0]):
            dataframe.loc[:, 'lat'][line] = np.zeros(number)

        dataframe['lon'] = np.zeros(dataframe.shape[0]).astype(object)

        for line in range(dataframe['lon'].shape[0]):
            dataframe.loc[:, 'lon'][line] = np.zeros(number)

        return dataframe


    def calc_dataframe_bins(self, dataframe): # ✓  
        bins_names = ['Latitude_bins', 'Longitude_bins']
        bins_file = "Bins_equidistant.txt"
        bins = pd.read_table(bins_file, sep='\t', names=bins_names, skipinitialspace=True)
        lat_bin, lon_bin = np.loadtxt("Bins_equidistant.txt", delimiter='\t', usecols=(0, 1), unpack=True)
        unique_lat = np.unique(np.array((bins["Latitude_bins"])))
        correspond_lon = []
        for i in range(len(unique_lat)):
            correspond_lon.append([])
        for i in range(len(bins)):
            idx = list(unique_lat).index(bins.iloc[i, 0])
            correspond_lon[idx].append(bins.iloc[i, 1])
        for line in range(dataframe.shape[0]):
            dis_down = self.calc_dis_down([dataframe['Elevation'][line]], input_unit="Deg")
            dis_up = self.calc_dis_up([dataframe['Elevation'][line]], input_unit="Deg")
            dis_step = self.calc_dis([dataframe['Elevation'][line]], input_unit="Deg")/3
            lat, lon, dis = self.obs_to_geo([dataframe['Elevation'][line]], [dataframe['Azimuth'][line]], np.arange(dis_down, dis_up, dis_step), dataframe['Latitude'][line], dataframe['Longitude'][line])
#             print(dis_up)
            for position in range(3):
                dataframe['lat'][line][position] = self.calc_close_bin_faster(lat_bin, unique_lat, correspond_lon, lat[position], lon[position])[0]
                dataframe['lon'][line][position] = self.calc_close_bin_faster(lat_bin, unique_lat, correspond_lon, lat[position], lon[position])[1]
        return dataframe

    
    def recalc_dataframe(self, dataframe, name_of_value, func = "mean"): # ✓  
        lat = []
        lon = []
        value = []

        for i in range(dataframe.shape[0]):
            for j in range(dataframe["lat"][0].shape[0]):
                lat_temp = dataframe["lat"][i][j]
                lon_temp = dataframe["lon"][i][j]
                lat.append(lat_temp)
                lon.append(lon_temp)
                value.append(dataframe[name_of_value][i])

        df = pd.DataFrame(data={name_of_value: value, 'lat': lat, "lon":lon})
        if func=="mean":
            df_temp1 = df.groupby(['lat', "lon"]).mean() ###function
        elif func=="max":
            df_temp1 = df.groupby(['lat', "lon"]).max()
        df_temp2 = df.groupby(['lat', "lon"]).count()
        df_temp1 = df_temp1.reset_index()
        df_temp2 = df_temp2.reset_index()
        df_temp1["count"] = df_temp2[name_of_value]
        return df_temp1

    
    def show_globe_map(self, dataframe, name_of_value, name_of_size, name_of_lat, name_of_lon, center=(0, 0), zoom=(45, 45), save=None): # ✓   
        colorscale = [
        [0, 'rgb(0, 0, 255)'], 
        [0.1, 'rgb(0,255,0)'],
        [0.2, 'rgb(255,255,0)'],
        [1, 'rgb(255, 0, 0)']
        ]


        fig = px.scatter_geo(
        dataframe,
        lat=name_of_lat,
        lon=name_of_lon,
        hover_name=name_of_value,
        color=name_of_value,
        color_continuous_scale=colorscale, #"thermal",
        size=name_of_value,
        size_max=10,
        width=1000,
        height=1000,
        range_color=[0, 1],
    )


        fig.update_layout(
        geo=dict(
            showland=True,
            landcolor="rgb(127, 127, 127)",
            subunitcolor="rgb(255, 0, 0)",
            countrycolor="rgb(0, 0, 0)",
            showlakes=True,
            lakecolor="rgb(125, 125, 255)",
            showsubunits=True,
            showcountries=True,
            resolution=50,
            projection=dict(
                type="orthographic",
                rotation_lon=center[1],
                rotation_lat=center[0],
            ),
            center=dict(lat=center[0], lon=center[1]),
            lonaxis=dict(showgrid=True, gridwidth=0.5, range=[center[1]-zoom[1], center[1]+zoom[1]], dtick=1),
            lataxis=dict(showgrid=True, gridwidth=0.5, range=[center[0]-zoom[0], center[0]+zoom[0]], dtick=1),
        ),
        title=f"Ionospheric Scintillation - centered (lat={center[0]}, lon={center[0]}) - zoom (lat={zoom[0]}, lon={zoom[0]}) - Parameter {name_of_value}",
    )
        if save:
            print("Ukladam")
            save_name = save
            if save==True:
                save_name = f"Ionospheric Scintillation - centered (lat={center[0]}, lon={center[0]}) - zoom (lat={zoom[0]}, lon={zoom[0]}) - Parameter {name_of_value}.png"
                print(save_name)
    #         fig.write_image(save_name, format='png',engine='kaleido')
    #         img_bytes = fig.to_image(format="png")
    #         im1 = img_bytes.save(save_name)

        fig.show()
    
    
if __name__ == '__main__':
    pass
