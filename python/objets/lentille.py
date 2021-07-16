class LentilleGravita():
    
    def __init__(self, ra_deg: float, dec_deg: float):
        self.ra = ra_deg
        self.dec = dec_deg
        self.seeing = float()
        self.exposition = float()
        
    def compute_seeing(self, list_of_w_objects):
        obj = self.closest_object(list_of_w_objects)
        return obj.seeing
    
    def compute_exposition(self, list_of_w_objects):
        obj = self.closest_object(list_of_w_objects)
        return obj.exposition
    
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
        
    def __str__(self):
        os = str()
        os += "Lentille: \n"
        os += f"RA: {self.ra} (deg)" + "\n"
        os += f"DEC: {self.dec} (deg)" + "\n"
        os += f"Exposition: {self.exposition} (min)" + "\n"
        os += f"Seeing: {self.seeing} (arc sec)" + "\n"
        return os
