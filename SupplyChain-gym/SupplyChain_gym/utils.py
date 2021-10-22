import numpy as np

def assign_env_config(self, kwargs):
    for key, value in kwargs.items():
        #print(f'Setting Attribute Outside for {key} with {value}')
        setattr(self, key, value)
    if hasattr(self, 'env_config'):
        # print(self.env_config)
        for key, value in self.env_config.items():
            #print('Setting Attribute Inside')
            # Check types based on default settings
            if hasattr(self, key):
                if type(getattr(self,key)) == np.ndarray:
                    setattr(self, key, value)
                    #print(f'Setting Attribute for Array {key} with {value}')
                elif type(getattr(self,key)) == tuple:
                    #print(f'Setting Attribute for Tuple {key} with {value[0]}')
                    setattr(self, key, value[0])
                    
                elif type(getattr(self,key)) == int:
                    #print(f'Setting Attribute for Int {key} with {value}')
                    setattr(self, key, value)
                else:
                    #print(f'Setting Attribute for Others {key} with {value}')
                    setattr(self, key,type(getattr(self, key))(value))
                    
                        
            else:
                setattr(self, key, value)
