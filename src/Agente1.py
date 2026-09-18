import math
import numpy as np

class HokuyoSensorSim(object):
    """
    Simulates a Hokuyo laser sensor in CoppeliaSim using vision sensors.

    This class provides an interface to interact with a simulated Hokuyo sensor,
    typically attached to a robot in CoppeliaSim. It manages the underlying vision
    sensors and provides methods to retrieve sensor data in either range or point format.

    Attributes:
        _sim: The simulation API object used to interact with CoppeliaSim.
        _base_name (str): The name of the base object to which the Hokuyo sensor is attached.
        _is_range_data (bool): Determines if sensor data is returned as range values (True) or 3D points (False).
        _base_obj: The handle of the base object in the simulation.
        _vision_sensors_obj (list): Handles of the vision sensors used to simulate the Hokuyo sensor.

    Args:
        sim: The simulation API object.
        base_name (str): The name of the base object (must contain 'fastHokuyo').
        is_range_data (bool, optional): If True, sensor data is returned as range values. Defaults to False.

    Raises:
        ValueError: If 'fastHokuyo' is not in the base_name, or if the base object or vision sensors are not found.

    Methods:
        get_is_range_data() -> bool:
            Returns whether sensor data is returned as range values.

        set_is_range_data(is_range_data: bool) -> None:
            Sets whether sensor data should be returned as range values.

        getSensorData():
            Retrieves sensor data from the vision sensors.
            Returns either a list of range values or a list of 3D points, depending on _is_range_data.
    """

    _sim = None

    _base_name = ""
    _vision_sensor_name_template = "{}/sensor{}"

    # _vision_sensors_obj will be initialized in __init__
    _base_obj = None
    _is_range_data = False
    
    _angle_min=-120*math.pi/180
    _angle_max=120*math.pi/180
    _angle_increment=(240/684)*math.pi/180 # angle: 240 deg, pts: 684

    def __init__(self, sim, base_name, is_range_data=True):
        self._sim = sim
        self._base_name = base_name
        self._is_range_data = is_range_data

        if "fastHokuyo" not in base_name:
            raise ValueError(
                f"ERR: fastHokuyo must be in the base object name. Ex: `/PioneerP3DX/fastHokuyo`"
            )

        self._base_obj = sim.getObject(base_name)
        if self._base_obj == -1:
            raise ValueError(
                f"ERR: base_obj ({self._base_obj}) is not a valid name in the simulation"
            )

        self._vision_sensors_obj = [
            sim.getObject(self._vision_sensor_name_template.format(self._base_name, 1)),
            sim.getObject(self._vision_sensor_name_template.format(self._base_name, 2)),
        ]

        if any(obj == -1 for obj in self._vision_sensors_obj):
            raise ValueError(
                f"ERR: the _vision_sensors_obj names are not valid in the simulation"
            )

    def get_is_range_data(self) -> bool:
        return self._is_range_data

    def set_is_range_data(self, is_range_data: bool) -> None:
        self._is_range_data = is_range_data

    def getSensorData(self):
        
        angle = self._angle_min
        sensor_data = []
        
        for vision_sensor in self._vision_sensors_obj:
            r, t, u = sim.readVisionSensor(vision_sensor)
            if u:
                sensorM = sim.getObjectMatrix(vision_sensor)
                relRefM = sim.getObjectMatrix(self._base_obj)
                relRefM = sim.getMatrixInverse(relRefM)
                relRefM = sim.multiplyMatrices(relRefM, sensorM)

                p = [0, 0, 0]
                p = sim.multiplyVector(sensorM, p)
                t = [p[0], p[1], p[2], 0, 0, 0]
                for j in range(int(u[1])):
                    for k in range(int(u[0])):
                        w = 2 + 4 * (j * int(u[0]) + k)
                        v = [u[w], u[w + 1], u[w + 2], u[w + 3]]
                        angle = angle + self._angle_increment
                        if self._is_range_data:
                            sensor_data.append([angle, v[3]])
                        else:
                            p = sim.multiplyVector(relRefM, v)
                            sensor_data.append([p[0], p[1], p[2]])
                            
        return np.array(sensor_data)
    
class Agent:
    def __init__(self, id_robo):
        self.id = id_robo
    
        self.robotname = f'PioneerP3DX[{self.id - 1}]'

        self.L = 0.381  # Metros
        self.r = 0.0975 # Metros
            
    def movement(self, sim, img, goal_list):

        ## Código placeholder, esse algoritmo não faz nada, apenas desvia para esquerda ou direita 
        ## Utilize esse código para entender como declarar as variáveis necessárias.

        # sim = objeto chave de conexão com o coppelia.
        # img = visualização da imagem do ambiente. Algoritmos deliberativos irão precisar disso.
        # goal_list = lista de pontos de interesse a serem visitados no formato [[x1, y1],[x2, y2],[x3, y3],...]

        ## O metodo deverá retornar TRUE ao alcançar um dos goals (FALSE caso contrário), lembre-se de chegar o mais perto possível antes de retornar.
        
        # Fazendo leitura do laser
        hokuyo_sensor = HokuyoSensorSim(sim, "/" + self.robotname + "/fastHokuyo")
        
        laser_data = hokuyo_sensor.getSensorData()

        l_wheel = sim.getObject('/' + self.robotname + '/leftMotor')
        r_wheel = sim.getObject('/' + self.robotname + '/rightMotor')
    
        # Velocidade básica (linear, angular)
        v = 0
        w = np.deg2rad(0)

        frente = int(len(laser_data) / 2)
        lado_direito = int(len(laser_data) * 1 / 4)
        lado_esquerdo = int(len(laser_data) * 3 / 4)

        # Lógica de desvio de obstáculo
        if laser_data[frente, 1] > 2:
            v = .5
            w = 0
        elif laser_data[lado_direito, 1] > 2:
            v = 0
            w = np.deg2rad(-30)
        elif laser_data[lado_esquerdo, 1] > 2:
            v = 0
            w = np.deg2rad(30)

        # Modelo cinemático
        wl = v / self.r - (w * self.L) / (2 * self.r)
        wr = v / self.r + (w * self.L) / (2 * self.r)

        # Enviando velocidades (não precisa mais de opmode)
        sim.setJointTargetVelocity(l_wheel, wl)
        sim.setJointTargetVelocity(r_wheel, wr)