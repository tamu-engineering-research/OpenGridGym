import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dss import DSS as dssObj
from ipywidgets import interact, Dropdown
from scipy.sparse import csr_matrix
import functools

def get_measurable_fields(obj):
    '''
    Returns list of fields accessible in active item in OpenDSS.
    For example, obj could be the active circuit element.
    
    Example:
    
        >>> get_measurable_fields(dss.Elem)
        ['AllPropertyNames', 'AllVariableNames', ...,
                                    'VoltagesMagAng', 'Yprim']
    
    '''
    fields = []
    for attr in dir(obj):
        value = getattr(obj, attr)
        if not callable(value) and not attr.startswith('_'):
            fields.append(attr)

    return fields

def case_updated(instance_method):
    '''
    A decorator for methods to trigger when a dss case
    gets updated.

    '''
    @functools.wraps(instance_method) # to preserve method signature
    def new_instance_method(self, *a, **kw):
        output = instance_method(self, *a, **kw)

        self.needs_saving = True

        return output

    return new_instance_method

def DSS_from_docstring(commands):
    '''
    Return a dss object corresponding to multiline OpenDSS commands
    provided in the form of a docstring.
    
    commands: str
        Docstring containing all the OpenDSS commands to be run as if
        it were a .dss file.
    
    '''
    with open('temp.dss', 'w') as file:
        file.write(commands)

    dss = DSS(file.name)

    os.remove(file.name)
    
    return dss

def DSS_from_empty(circuit_name='default_circuit', basekv=115.0):
    '''
    Return a dss object corresponding to an empty circuit with default
    parameters and no elements (other than default Vsource).
    
    circuit_name: str
        Circuit name used when creating empty circuit, without spaces.

    basekv: float
        Circuit's base voltage in kV.
    
    '''
    assert ' ' not in circuit_name, 'circuit_name cannot contain spaces.'

    commands = f'''

        Clear

        New Circuit.{circuit_name} basekv={basekv}

        Set VoltageBases=[{basekv}]
        CalcVoltageBases

        Solve
    '''
    
    return DSS_from_docstring(commands)

class DSS:
    '''
    A class to interface with OpenDSS.

    case_path: str
        Path to '.dss' case file. If name of file is provided without
        full path, current directory is assumed.

    Example:
        
        >>> case_path = 'some_case.dss'
        >>> dss = DSS(case_path)

    '''
    def __init__(self, case_path):
        self.case_path = case_path
        self._check_case_path()

        starting_directory = os.getcwd()
        try:
            self._initialize_OpenDSS()
            self._compile_case()
            self._set_data_path(starting_directory)
            self.solve_case()
            self.update_all_element_names()
            self._avoid_exception_15013()
            os.chdir(starting_directory)
        except:
            # In case above fails, you don't want OpenDSS to send the
            # current working directory to your Documents folder.
            # Instead, you bring it back to where it was previously.
            os.chdir(starting_directory)
            raise NotImplementedError('Could not initialize OpenDSS case or '\
                                        'could not compile or solve. Needs further review.')
        
    def __repr__(self):
        NumBuses = self.Circuit.NumBuses
        NumNodes = self.Circuit.NumNodes
        return f'DSS(NumBuses={NumBuses}, NumNodes={NumNodes})'

    @classmethod
    def new_case_from_docstring(cls, commands):
        '''
        Return a dss object corresponding to multiline OpenDSS commands
        provided in the form of a docstring.
        
        commands: str
            Docstring containing all the OpenDSS commands to be run as if
            it were a .dss file.
        
        '''
        # Ensure all is first cleared
        commands = 'Clear\n' + commands

        with open('temp.dss', 'w') as file:
            file.write(commands)

        dss = cls.open_case_from_file(file.name)

        os.remove(file.name)
        
        return dss

    @classmethod
    def new_case_from_empty(cls, circuit_name='default_circuit', basekv=115.0):
        '''
        Return a dss object corresponding to an empty circuit with default
        parameters and no elements (other than default Vsource).
        
        circuit_name: str
            Circuit name used when creating empty circuit, without spaces.

        basekv: float
            Circuit's base voltage in kV.
        
        '''
        assert ' ' not in circuit_name, 'circuit_name cannot contain spaces.'

        commands = f'''

            Clear

            New Circuit.{circuit_name} basekv={basekv}

            Set VoltageBases=[{basekv}]
            CalcVoltageBases

            Solve
        '''
        
        return cls.new_case_from_docstring(commands)

    @classmethod
    def open_case_from_file(cls, case_path):
        '''
        Open an existing case file given path (.dss).

        '''

        return cls(case_path)

    @classmethod
    def open_case_from_folder(cls, folder):
        '''
        Open an existing case file given folder.
        This assumes Master.dss exists and is the main
        file in the folder.

        '''
        master_path = os.path.join(folder, 'Master.dss')

        if os.path.isfile(master_path):

            return cls.open_case_from_file(master_path)

        else:

            raise ValueError(f'No Master.dss file found in {folder}. '
                    f'You may use {cls.__name__}.open_case_from_file instead.')

    def _check_case_path(self):
        '''
        Ensure that self.case_path is a valid path
        to an OpenDSS case file.

        '''
        # Force full path and check if it's a file
        self.case_path = os.path.abspath(self.case_path)
        if not os.path.isfile(self.case_path):
            raise ValueError(f'The following path does not lead to a file:\n{self.case_path}')

        # Check for the '.dss' extension (OpenDSS)
        if self.case_path[-4:] != '.dss':
            raise ValueError(f'Expected an OpenDSS file (.dss). Instead, got:\n{self.case_path}')

    def _set_data_path(self, data_path):
        '''
        Sets data path so that OpenDSS knows where
        the defualt path is (e.g. for saving purposes).

        '''
        self.Obj.DataPath = data_path
        if self.Obj.DataPath != data_path:
            return ValueError(f'''
                Requested the following as new data path:
                    {data_path}
                but the following is still the data path:
                    {self.Obj.DataPath}
                ''')
        
    def _initialize_OpenDSS(self):
        '''
        Initialize useful interfaces with OpenDSS.

        '''
        self.Obj = dssObj
        self.Text = self.Obj.Text
        self.Circuit = self.Obj.ActiveCircuit
        self.Solution = self.Circuit.Solution
        self.Elem = self.Circuit.ActiveCktElement
        self.Bus = self.Circuit.ActiveBus
        
    def _compile_case(self):
        '''
        Instruct OpenDSS to compile case given in self.case_path.

        '''
        result = self.command(f"compile [{self.case_path}]")

        self.NumBuses = self.Circuit.NumBuses
        self.NumNodes = self.Circuit.NumNodes

        self._create_node_to_bus_matrix()

        return result or None
        
    def solve_case(self):
        '''
        Instruct OpenDSS to solve power flow.

        '''
        # self.command('Solve') # alternative
        result = self.Solution.Solve()

        if (self.NumBuses != self.Circuit.NumBuses) or \
           (self.NumNodes != self.Circuit.NumNodes):

            self._create_node_to_bus_matrix()

            self.NumBuses = self.Circuit.NumBuses
            self.NumNodes = self.Circuit.NumNodes

        return result or None
    
    def update_all_element_names(self):
        '''
        Creates/updates self.element_names dict where
        each key is a type of element (e.g. 'Generator') and
        each value is a list of **full** element names.
        
        Example:
        
            >>> dss.update_all_element_names()
            >>> dss.element_names['Generator']
            ['Generator.gen_1', 'Generator.gen_2']
        
        '''
        self.element_names = {}
        for elem_name in self.Circuit.AllElementNames:
            
            # Pull element class (e.g. Generator or Load)
            class_name = elem_name.split('.')[0]
            
            # If never seen this class before, initialize list
            if class_name not in self.element_names.keys():
                self.element_names[class_name] = []
                
            # Add element names to this class' list
            self.element_names[class_name].append(elem_name)

    def _avoid_exception_15013(self):
        '''
        INTERNAL: Temporary solution to a problem experienced with
        dss_python exception:

        "DSSException: (#15013) Nodes are not initialized. Try solving the system first."

        '''
        for elem_name in self.Circuit.AllElementNames:
            self.Circuit.SetActiveElement(elem_name)

    def _create_node_to_bus_matrix(self):
        '''
        Creates a sparse matrix 'N2B' that's used to very quickly
        extract bus voltages from node voltages.

        Used internally.

        '''
        N2B = np.zeros((self.Circuit.NumBuses, self.Circuit.NumNodes))
        for i, elem in enumerate(self.iterate('Bus')):
            bus = elem.Name
            num_nodes = elem.NumNodes
            for j, node in enumerate(self.Circuit.AllNodeNames):
                
                bus_at_node, node_idx = node.split('.')
                if bus_at_node != bus:
                    continue
                
                node_idx = int(node_idx)
                if num_nodes < 1:
                    raise NotImplementedError('Unfamiliar with less than 1 node at bus.')
                elif num_nodes == 1:
                    N2B[i, j] = 1.0
                elif num_nodes == 2:
                    if node_idx < 1:
                        raise NotImplementedError('')
                    elif node_idx in (1, 2):
                        N2B[i, j] = 1/2
                elif num_nodes >= 3:
                    if node_idx < 1:
                        raise NotImplementedError('')
                    elif node_idx in (1, 2, 3):
                        N2B[i, j] = 1/3

        self.N2B = csr_matrix(N2B)

    @case_updated
    def _new_element(self, class_name, elem_name, **prop_kwargs):
        '''
        A helper method used to create new elements to avoid
        repeating redundant code.

        '''
        command = f'New {class_name}.{elem_name}'
        for prop, val in prop_kwargs.items():
            command += f' {prop}={val}'
        
        self.command(command)
        self.solve_case()
        self.update_all_element_names()

    def new_load(self, load_name, bus, **prop_kwargs):
        '''
        Creates a new load given required load_name and bus.

        load_name: str
            Name of load (excluding 'Load.')

        bus: str
            Bus name where load is to be connected

        prop_kwargs: dict
            Properties to be passed directly to OpenDSS

        '''
        prop_kwargs['bus1'] = bus
        
        for prop in ('kW', 'kvar'):
            if prop not in prop_kwargs:
                prop_kwargs[prop] = 0.0

        self._new_element('Load', load_name, **prop_kwargs)

    def new_generator(self, generator_name, bus, **prop_kwargs):
        '''
        Creates a new generator given required generator_name and bus.

        generator_name: str
            Name of generator (excluding 'Generator.')

        bus: str
            Bus name where generator is to be connected

        prop_kwargs: dict
            Properties to be passed directly to OpenDSS

        '''
        prop_kwargs['bus1'] = bus
        
        for prop in ('kW', 'kvar'):
            if prop not in prop_kwargs:
                prop_kwargs[prop] = 0.0

        self._new_element('Generator', generator_name, **prop_kwargs)

    def new_line(self, line_name, bus_from, bus_to, **prop_kwargs):
        '''
        Creates a new line given required line_name, bus_from and bus_to.

        line_name: str
            Name of line (excluding 'Line.')

        bus_from: str
            Bus name where line is to be connected (FROM)

        bus_to: str
            Bus name where line is to be connected (TO)

        prop_kwargs: dict
            Properties to be passed directly to OpenDSS

        '''
        prop_kwargs['bus1'] = bus_from
        prop_kwargs['bus2'] = bus_to

        self._new_element('Line', line_name, **prop_kwargs)

    def get_node_df(self):
        '''
        Returns observation dataframe, where index is unique node names.

        '''
        Vbase = (self.Circuit.AllBusVmag / self.Circuit.AllBusVmagPu).reshape(-1, 1)
        Vri = self.Circuit.AllBusVolts.reshape(-1,2) / Vbase
        VmagPu = self.Circuit.AllBusVmagPu.reshape(-1,1)
        node_df = np.concatenate((Vbase, Vri, VmagPu), axis=1)
        node_df = pd.DataFrame(node_df, index=self.Circuit.AllNodeNames, columns=['Vbase', 'Vr', 'Vi', 'VmagPu'])
        return node_df

    def get_bus_df(self):
        '''
        Returns observation dataframe, where index is unique bus names.

        '''
        VmagPu = (self.N2B @ self.Circuit.AllBusVmagPu).reshape(-1, 1)
        bus_xy = [[bus.x, bus.y] for bus in self.iterate('Bus')]
        bus_df = np.concatenate((bus_xy, VmagPu), axis=1)
        bus_df = pd.DataFrame(bus_df, index=self.Circuit.AllBusNames, columns=['x', 'y', 'VmagPu'])
        return bus_df

    def get_vsource_df(self):
        '''
        Returns observation dataframe, where index is unique vsource names.

        '''
        vsource_dict = {'enabled': [], 'bus': [], 'kW': [], 'kvar': []}

        if self.Circuit.Vsources.Count > 0:
            index = self.Circuit.Vsources.AllNames
            for vsource in self.iterate('Vsource'):
                
                vsource_dict['enabled'].append(vsource.Enabled)
                
                bus = vsource.BusNames[0].split('.')[0]
                vsource_dict['bus'].append(bus)
                
                kW, kvar = vsource.TotalPowers[:2]
                vsource_dict['kW'].append(kW)
                vsource_dict['kvar'].append(kvar)
        else:
            index = []

        vsource_df = pd.DataFrame(vsource_dict, index=index)
        return vsource_df

    def _get_prosumer_group_df(self, class_name, circuit_obj, sign=+1.0):
        '''
        Returns observation dataframe, where index is unique load/gen names.

        Used internally. Refer to 'get_load_df' for an example.

        sign: float/int
            +1 for default consumers (e.g. load) and
            -1 for default producers (e.g. generators)

        '''
        elem_dict = {'enabled': [], 'bus': [], 'kW': [], 'kvar': []}

        if circuit_obj.Count > 0:
            index = circuit_obj.AllNames
            for elem in self.iterate(class_name):
                
                elem_dict['enabled'].append(elem.Enabled)
                
                bus = elem.BusNames[0].split('.')[0]
                elem_dict['bus'].append(bus)
                
                kW, kvar = sign * elem.TotalPowers
                elem_dict['kW'].append(kW)
                elem_dict['kvar'].append(kvar)
        else:
            index = []

        elem_df = pd.DataFrame(elem_dict, index=index)
        return elem_df

    def get_load_df(self):
        '''
        Returns observation dataframe, where index is unique load names.

        '''
        return self._get_prosumer_group_df('Load', self.Circuit.Loads, sign=+1.0)

    def get_generator_df(self):
        '''
        Returns observation dataframe, where index is unique gen names.

        '''
        return self._get_prosumer_group_df('Generator', self.Circuit.Generators, sign=-1.0)

    def _get_branch_group_df(self, class_name, circuit_obj):
        '''
        Returns observation dataframe, where index is unique branch names.

        '''
        elem_dict = {
                        'enabled': [],  'phases': [],
                        'bus_from': [], 'bus_to': [],
                        'max_amps': [],
                        'kW_from': [], 'kvar_from': [],
                        'kW_to': [], 'kvar_to': [],
                        'kW_loss': [], 'kvar_loss': [],
                        'flow_direction': [],
                     }

        if circuit_obj.Count > 0:
            index = circuit_obj.AllNames
            for elem in self.iterate(class_name):
                
                elem_dict['enabled'].append(elem.Enabled)
                elem_dict['phases'].append(elem.NumPhases)
                
                f = elem.BusNames[0].split('.')[0]
                t = elem.BusNames[1].split('.')[0]
                elem_dict['bus_from'].append(f)
                elem_dict['bus_to'].append(t)

                max_amps = elem.CurrentsMagAng[::2].max()
                elem_dict['max_amps'].append(max_amps)
                
                if elem.Enabled:
                    kW_from, kvar_from, _kW_to, _kvar_to = elem.TotalPowers
                else:
                    kW_from, kvar_from, _kW_to, _kvar_to = 0., 0., 0., 0.
                elem_dict['kW_from'].append(kW_from)
                elem_dict['kvar_from'].append(kvar_from)
                elem_dict['kW_to'].append(-_kW_to)
                elem_dict['kvar_to'].append(-_kvar_to)
                
                kW_loss, kvar_loss = elem.Losses / 1000
                elem_dict['kW_loss'].append(kW_loss)
                elem_dict['kvar_loss'].append(kvar_loss)

                sign = np.sign(np.abs(kW_from) - np.abs(_kW_to))
                elem_dict['flow_direction'].append(sign)
        else:
            index = []

        elem_df = pd.DataFrame(elem_dict, index=index)
        return elem_df

    def get_line_df(self):
        '''
        Returns observation dataframe, where index is unique line names.

        '''
        return self._get_branch_group_df('Line', self.Circuit.Lines)

    def get_transformer_df(self):
        '''
        Returns observation dataframe, where index is unique transformer names.

        '''
        return self._get_branch_group_df('Transformer', self.Circuit.Transformers)

    def get_all_branches_df(self):
        '''
        Returns observation dataframe, where index is unique branch names.

        This combines all branches (e.g. lines & transformers)

        '''
        line_df = self.get_line_df()
        trans_df = self.get_transformer_df()

        line_df.index = line_df.index.map(lambda name: f'Line.{name}')
        trans_df.index = trans_df.index.map(lambda name: f'Transformer.{name}')

        branch_df = pd.DataFrame()
        branch_df = branch_df.append([line_df, trans_df])

        return branch_df

    def save_case_as(self, directory=None, overwrite=False):
        '''
        NEEDS REVIEW!

        TEMPORARY NOTES:

            Given a ***directory**, save the entire case using
            OpenDSS' Save command (creates and organizes dss files
            in one folder).

            If user specifies a directory that already exists, either raise
            error and suggest that they should specify 'overwrite=True' or
            ignore and overwrite if 'overwrite=True'.

            If user does not specify directory, use circuit name followed
            by time now from year to microsecond (6 digits after second).

            Correspondingly, when "opening" an existing case, the user now
            has to specify a '.dss' file (e.g. Master) but for consistency,
            if the user specifies a folder instead, it should by default
            assume that 'Master.DSS' is the main one. For this, you need to
            ensure there exists such a file in the directory.

            PROBLEM: If you save into a directory that exists (and allow
            overwriting), and the directory already has some files in there
            from a previous case, they will stay there. Ideally, you'd like
            to erase the entire folder first. However, you might have some
            results that are useful to you. What if you just remove '.dss'
            files? Even then, you might have some useful files for you. I
            don't have a solution at the moment, but it seems like the way
            to go is to advise the user not overwrite to existing folder.

        '''
        if directory is None:
            
            time_now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')

            directory = f'{self.Circuit.Name}_{time_now}'

        if os.path.isdir(directory) and not overwrite:
            
            raise ValueError(f'Specified directory already exists. '
                f'To overwrite, set keyword argument overwrite=True.')

        save_command = f'Save Circuit dir="{directory}"'

        result = self.command(save_command)

        self.needs_saving = False

        return result or None

    def delete_case(self, directory=None):
        '''
        Placeholder to remind user not to automatically delete a case file!

        '''
        raise ValueError(f'Please delete manually by deleting the directory '
            f'containing the case, but first ensure that no other useful '
            f'files or folders are contained within it!')
    
    def summarize(self):
        '''
        Prints a detailed summary of the circuit.

        '''
        # print('Case path:', self.case_path)
        
        print(self.Circuit.NumBuses, 'Buses')
        print(self.Circuit.NumNodes, 'Nodes')

        print('Voltage Bases (kV L-L):\t', ', '.join(map(str, self.Circuit.Settings.VoltageBases)))

        print('Circuit Elements:')
        for class_name, names in self.element_names.items():
            count = len(names)
            print(f'\t\t{count} {class_name}{"s"*(count > 1)}')
        
    def real_imag_to_complex(self, real_imag):
        '''
        Map OpenDSS representation of complex arrays to a more
        familiar tall complex vector, as illustrated below.

        real_imag: tuple
            OpenDSS representation of complex arrays (see below).

        Example:

            Input (real_imag):
            
                # tuple of shape (2*n,)
                (r1, x1, r2, x2, ..., rn, xn)
                
            Output:
            
                # complex numpy array of shape (n, 1)
                [[r1 + 1j*x1,
                  r2 + 1j*x2,
                  ...
                  rn + 1j*xn]]
        
        '''
        return np.array(real_imag).reshape(-1,2) @ np.array([[1], [1j]])
        
    @case_updated
    def edit(self, elem_name, **props):
        '''
        Edit any element's properties.

        elem_name: str
            Full circuit element name (e.g. 'Generator.gen_1').

        props: case-insensitive kwargs
            Since OpenDSS is case-insensitive, these keyword
            arguments are passed as is.
            
        Note: even if no 'props' were provided, OpenDSS still
        accepts an empty edit command e.g. 'Edit Load.load_1'.
        
        Example:
            
            # Edit some generator's PQ setpoints
            >>> dss.edit('Generator.gen_1', kW=50, kvar=23)
            
            # Edit nothing about the generator
            >>> dss.edit('Generator.gen_1')
        
        '''
        # Check if element in circuit
        if elem_name not in self.Circuit.AllElementNames:
            if elem_name.lower() not in map(lambda s: s.lower(), self.Circuit.AllElementNames):
                raise ValueError(f'Element {elem_name} not found in circuit.')

        # Specify command to OpenDSS
        command = f'Edit {elem_name}'
        for prop, val in props.items():
            command += f' {prop}={val}'
            
        # Execute command
        result = self.command(command)

        # Return error if any
        return result or None

    def read(self, elem_name, prop):
        '''
        Reads an element's property from OpenDSS directly.

        elem_name: str
            Unique element name (as it appears in self.Circuit.AllElementNames)

        prop: str
            Name of property being inspected

        '''
        result = self.command(f'? {elem_name}.{prop}')
        
        if result == 'Property Unknown':
            raise ValueError(f'Property "{prop}" of element "{elem_name}" is invalid.')
        else:
            return result

    def command(self, command_str):
        '''
        Sends argument as a command to OpenDSS, and returns result.

        Result is an empty string (bool is False) if no OpenDSS 'errors'.

        command_str: str
            Command to be sent directly to OpenDSS

        '''
        if not isinstance(command_str, str):
            raise TypeError(f'Command expected to be str, '
                            f'not {type(command_str)}.')

        self.Text.Command = command_str

        return self.Text.Result

    def command_multiline(self, command_docstring):
        '''
        Equivalent to self.command, but argument is a docstring.

        command_docstring: str
            Multi-line string (docstring). Each line will be sent
            from top to bottom and a dictionary of error messages
            will be returned.

        '''
        non_empty_results = {}
        for command in command_docstring.splitlines():
            result = self.command(command)
            if result:
                non_empty_results[command] = result

        return non_empty_results
        
    def iterate(self, class_name):
        '''
        Returns an iterator which yields "CktElement Interface" COM objects
        (OpenDSS terminology) corresponding to circuit elements of the same
        type, specified by class_name.

        class_name: str
            One of the keys of self.element_names representing
            a type of element (e.g. 'Line').

        Example:

            # Print From & To buses of all lines
            >>> for elem in dss.iterate('Line'):
            ...     print(elem.Name, elem.BusNames)

        '''
        if class_name == 'Bus':
            for bus_name in self.Circuit.AllBusNames:
                yield self.Circuit.Buses(bus_name)
        else:
            if class_name not in self.element_names.keys():
                raise ValueError(f'{class_name} is an invalid class name. '\
                                 f'Select from these:\n{[*self.element_names.keys()] + ["Bus"]}')

            for elem_name in self.element_names[class_name]:
                yield self.Circuit.CktElements(elem_name)
            
    def element_data(self, class_name='Vsource', field='Voltages'):
        '''
        For use in JupyterLab or JupyterNotebook.
        
        Produce an interactive set of dropdowns that help you
        observe element properties and measurements (post-solution).
        
        Result can then be extracted from self.interact_result.
        
        '''
        element_names = {**self.element_names,
                         **{'Bus': self.Circuit.AllBusNames}}

        self.interact_result = None
        @interact(class_name=Dropdown(options=element_names.keys(), value=class_name))
        def _(class_name):
            obj = self.Bus if class_name == 'Bus' else self.Elem
            @interact(elem_name=Dropdown(options=element_names[class_name]),
                     field=Dropdown(options=get_measurable_fields(obj), value=field))
            def _(elem_name, field):
                if class_name == 'Bus':
                    self.Circuit.SetActiveBus(elem_name)
                    self.interact_result = getattr(self.Bus, field)
                else:
                    self.Circuit.SetActiveElement(elem_name)
                    self.interact_result = getattr(self.Elem, field)
                
                display(self.interact_result)
            
    def element_props(self, class_name='Vsource'):
        '''
        For use in JupyterLab or JupyterNotebook.
        
        Produce an interactive set of dropdowns that help you
        observe element properties and measurements (pre-solution).
        
        Result can then be extracted from self.interact_result.
        
        '''
        self.interact_result = None
        @interact(class_name=Dropdown(options=self.element_names.keys(), value=class_name))
        def _(class_name):
            # initial
            elem_name = self.element_names[class_name][0]
            self.Circuit.SetActiveElement(elem_name)
            
            @interact(elem_name=Dropdown(options=self.element_names[class_name]),
                     prop=Dropdown(options=self.Elem.AllPropertyNames))
            def _(elem_name, prop):
                self.Circuit.SetActiveElement(elem_name)
                self.interact_result = self.Elem.Properties(prop).Val

                print()
                display(self.interact_result)
                # print()
                # print('='*20, 'DESCRIPTION', '='*20)
                # print()
                # print(self.Elem.Properties(prop).Description) # Seems to fail post dss_python 0.10

    def get_props_all_elements(self, class_name, numbers_as_floats=False):
        '''
        Return a dataframe showing properties of all elements of a certain type.

        class_name: str
            OpenDSS element type (e.g. 'Load' or 'Line')

        numbers_as_floats: bool
            If True, properties in dataframe are converted to floats.
            Otherwise, they remain as strings.

        '''
        for elem in self.iterate(class_name):
            break
        AllPropertyNames = elem.AllPropertyNames

        props_dict = {prop: [] for prop in AllPropertyNames}
        props_dict['Name'] = []
        for elem in self.iterate(class_name):
            props_dict['Name'].append(elem.Name.split('.')[1])
            for prop in AllPropertyNames:
                props_dict[prop].append(elem.Properties(prop).Val)

        props_df = pd.DataFrame(props_dict)
        props_df = props_df.set_index('Name')

        if numbers_as_floats:
            for prop in props_df:
                try:
                    props_df[prop] = props_df[prop].astype('float')
                except:
                    pass
        
        return props_df

    def get_props_single_element(self, elem_name, numbers_as_floats=False):
        '''
        Return a dataframe showing properties of a single element.

        elem_name: str
            Unique element name (as it appears in self.Circuit.AllElementNames)

        numbers_as_floats: bool
            If True, properties in dataframe are converted to floats.
            Otherwise, they remain as strings.

        '''
        self.Circuit.SetActiveElement(elem_name)
        AllPropertyNames = self.Elem.AllPropertyNames

        prop_dict = {}
        for prop in AllPropertyNames:
            val = self.Elem.Properties(prop).Val

            if numbers_as_floats:
                try:
                    val = float(val)
                except:
                    pass

            prop_dict[prop] = val
        
        return prop_dict