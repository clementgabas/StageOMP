class LentilleGravita():
    
    def __init__(self, ra_deg: float, dec_deg: float, field: str = str(), z_redshift: float = float(), arc_radius: float() = float()):
        self.ra = ra_deg
        self.dec = dec_deg
        self.w = field
        self.z_redshift = z_redshift
        self.z = self.z_redshift # alias
        self.arc_radius = arc_radius
        self.rad = self.arc_radius # alias
        
        self.seeing = float()
        self.exposition = float()
        self.ppv = None
        
    def compute_seeing(self, list_of_w_objects):
        obj = self.closest_object(list_of_w_objects)
        return obj.seeing
    
    def compute_exposition(self, list_of_w_objects):
        obj = self.closest_object(list_of_w_objects)
        return obj.exposition
    
    def compute_cadran(self, list_of_w_objects):
        obj = self.closest_object(list_of_w_objects)
        return obj.w
    
    def distance(self, other):
        out = ((self.ra - other.ra)**2 + (self.dec - other.dec)**2)**(1/2)
        return out
    
    def closest_object(self,  list_of_w_objects):
        current_min_dist = -1
        current_closest_objet = None
        for objet in list_of_w_objects:
            dist = self.distance(objet)
            if current_min_dist < 0:
                current_min_dist = dist
                current_closest_objet = objet
            else:
                if dist < current_min_dist:
                    current_min_dist = dist
                    current_closest_objet = objet
        return current_closest_objet

    def set_seeing(self, value: float):
        self.seeing = value
        
    def set_exposition(self, value: float):
        self.exposition = value
        
    def set_cadran(self, value: str):
        self.w = value
        
    def __str__(self):
        os = str()
        os += f"Lentille: ({self.w}) \n"
        os += f"Right Assent: {self.ra} (deg)" + "\n"
        os += f"Decli: {self.dec} (deg)" + "\n"
        os += f"RedShift: {self.z}" + "\n"
        os += f"Exposition: {self.exposition} (min)" + "\n"
        os += f"Seeing: {self.seeing} (arc sec)" + "\n"
        return os
    
    def set_ppv(self, other):
        self.ppv = other
        
