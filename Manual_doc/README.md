# Documentation
## Table of Contents

* [paraview vtk](#paraview-vtk)
    * [numpy_interface](#numpy_interface)
      * [dataset_adapter](#dataset_adapter)
      * [algorithms](#algorithms)
      * [internal_algorithms](#internal_algorithms)
    * [util](#util)
      * [colors](#colors)
      * [keys](#keys)
      * [misc](#misc)
      * [numpy_support](#numpy_support)
      * [vtkAlgorithm](#vtkAlgorithm)
      * [vtkConstants]
      * [vtkImageExportToArray]
      * [vtkImageExportFromArray]
      * [vtkMethodParser]
      * [vtkVariant]
    * [gtk](#getting-started)
      * [GtkGLExtVTKRenderWindow]
      * [GtkGLExtVTKRenderWindowInteractor]
      * [GtkVTKRenderWindow]
      * [GtkVTKRenderWindowInteractor]
    * [tk]
    * [web]
    * [wx]

## paraview vtk
import paraview.vtk

### numpy_interface
import paraview.vtk.numpy_interface
#### dataset_adapter
NAME
    paraview.vtk.numpy_interface.dataset_adapter

DESCRIPTION
    This module provides classes that allow Numpy-type access
    to VTK datasets and arrays. This is best described with some examples.
    
    To normalize a VTK array:
    
    from vtkmodules.vtkImagingCore vtkRTAnalyticSource
    import vtkmodules.numpy_interface.dataset_adapter as dsa
    import vtkmodules.numpy_interface.algorithms as algs
    
    rt = vtkRTAnalyticSource()
    rt.Update()
    image = dsa.WrapDataObject(rt.GetOutput())
    rtdata = image.PointData['RTData']
    rtmin = algs.min(rtdata)
    rtmax = algs.max(rtdata)
    rtnorm = (rtdata - rtmin) / (rtmax - rtmin)
    image.PointData.append(rtnorm, 'RTData - normalized')
    print image.GetPointData().GetArray('RTData - normalized').GetRange()
    
    To calculate gradient:
    
    grad= algs.gradient(rtnorm)
    
    To access subsets:
    
    >>> grad[0:10]
    VTKArray([[ 0.10729134,  0.03763443,  0.03136338],
           [ 0.02754352,  0.03886006,  0.032589  ],
           [ 0.02248248,  0.04127144,  0.03500038],
           [ 0.02678365,  0.04357527,  0.03730421],
           [ 0.01765099,  0.04571581,  0.03944477],
           [ 0.02344007,  0.04763837,  0.04136734],
           [ 0.01089381,  0.04929155,  0.04302051],
           [ 0.01769151,  0.05062952,  0.04435848],
           [ 0.002764  ,  0.05161414,  0.04534309],
           [ 0.01010841,  0.05221677,  0.04594573]])
    
    >>> grad[:, 0]
    VTKArray([ 0.10729134,  0.02754352,  0.02248248, ..., -0.02748174,
           -0.02410045,  0.05509736])
    
    All of this functionality is also supported for composite datasets
    even though their data arrays may be spread across multiple datasets.
    We have implemented a VTKCompositeDataArray class that handles many
    Numpy style operators and is supported by all algorithms in the
    algorithms module.
    
    This module also provides an API to access composite datasets.
    For example:
    
    from vtkmodules.vtkCommonDataModel import vtkMultiBlockDataSet
    mb = vtkMultiBlockDataSet()
    mb.SetBlock(0, image.VTKObject)
    mb.SetBlock(1e, image.VTKObject)
    cds = dsa.WrapDataObject(mb)
    for block in cds:
        print block
    
    Note that this module implements only the wrappers for datasets
    and arrays. The classes implement many useful operators. However,
    to make best use of these classes, take a look at the algorithms
    module.

CLASSES
    builtins.object
        ArrayAssociation
        CompositeDataIterator
            MultiCompositeDataIterator
        CompositeDataSetAttributes
        VTKCompositeDataArray
        VTKNoneArray
        VTKObjectWrapper
            DataObject
                CompositeDataSet
                DataSet
                    PointSet
                        PolyData
                        UnstructuredGrid
                Graph
                Molecule
                Table
            DataSetAttributes
    builtins.type(builtins.object)
        VTKArrayMetaClass
        VTKCompositeDataArrayMetaClass
        VTKNoneArrayMetaClass
    numpy.ndarray(builtins.object)
        VTKArray
    
    class ArrayAssociation(builtins.object)
     |  Easy access to vtkDataObject.AttributeTypes
     |  
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  CELL = 1
     |  
     |  FIELD = 2
     |  
     |  POINT = 0
     |  
     |  ROW = 6
    
    class CompositeDataIterator(builtins.object)
     |  CompositeDataIterator(cds)
     |  
     |  Wrapper for a vtkCompositeDataIterator class to satisfy
     |  the python iterator protocol. This iterator iterates
     |  over non-empty leaf nodes. To iterate over empty or
     |  non-leaf nodes, use the vtkCompositeDataIterator directly.
     |  
     |  Methods defined here:
     |  
     |  __getattr__(self, name)
     |      Returns attributes from the vtkCompositeDataIterator.
     |  
     |  __init__(self, cds)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __iter__(self)
     |  
     |  __next__(self)
     |  
     |  next(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class CompositeDataSet(DataObject)
     |  CompositeDataSet(vtkobject)
     |  
     |  A wrapper for vtkCompositeData and subclasses that makes it easier
     |  to access Point/Cell/Field data as VTKCompositeDataArrays. It also
     |  provides a Python type iterator.
     |  
     |  Method resolution order:
     |      CompositeDataSet
     |      DataObject
     |      VTKObjectWrapper
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  GetAttributes(self, type)
     |      Returns the attributes specified by the type as a
     |      CompositeDataSetAttributes instance.
     |  
     |  GetCellData(self)
     |      Returns the cell data as a DataSetAttributes instance.
     |  
     |  GetFieldData(self)
     |      Returns the field data as a DataSetAttributes instance.
     |  
     |  GetNumberOfCells(self)
     |      Returns the total number of cells of all datasets
     |      in the composite dataset. Note that this traverses the
     |      whole composite dataset every time and should not be
     |      called repeatedly for large composite datasets.
     |  
     |  GetNumberOfElements(self, assoc)
     |      Returns the total number of cells or points depending
     |      on the value of assoc which can be ArrayAssociation.POINT or
     |      ArrayAssociation.CELL.
     |  
     |  GetNumberOfPoints(self)
     |      Returns the total number of points of all datasets
     |      in the composite dataset. Note that this traverses the
     |      whole composite dataset every time and should not be
     |      called repeatedly for large composite datasets.
     |  
     |  GetPointData(self)
     |      Returns the point data as a DataSetAttributes instance.
     |  
     |  GetPoints(self)
     |      Returns the points as a VTKCompositeDataArray instance.
     |  
     |  __init__(self, vtkobject)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __iter__(self)
     |      Creates an iterator for the contained datasets.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  CellData
     |      This property returns the cell data of a dataset.
     |  
     |  FieldData
     |      This property returns the field data of a dataset.
     |  
     |  PointData
     |      This property returns the point data of the dataset.
     |  
     |  Points
     |      This property returns the points of the dataset.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from VTKObjectWrapper:
     |  
     |  __getattr__(self, name)
     |      Forwards unknown attribute requests to VTK object.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from VTKObjectWrapper:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class CompositeDataSetAttributes(builtins.object)
     |  CompositeDataSetAttributes(dataset, association)
     |  
     |  This is a python friendly wrapper for vtkDataSetAttributes for composite
     |  datsets. Since composite datasets themselves don't have attribute data, but
     |  the attribute data is associated with the leaf nodes in the composite
     |  dataset, this class simulates a DataSetAttributes interface by taking a
     |  union of DataSetAttributes associated with all leaf nodes.
     |  
     |  Methods defined here:
     |  
     |  GetArray(self, idx)
     |      Given a name, returns a VTKCompositeArray.
     |  
     |  PassData(self, other)
     |      Emulate PassData for composite datasets.
     |  
     |  __getitem__(self, idx)
     |      Implements the [] operator. Accepts an array name.
     |  
     |  __init__(self, dataset, association)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  append(self, narray, name)
     |      Appends a new array to the composite dataset attributes.
     |  
     |  keys(self)
     |      Returns the names of the arrays as a list.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class DataObject(VTKObjectWrapper)
     |  DataObject(vtkobject)
     |  
     |  A wrapper for vtkDataObject that makes it easier to access FielData
     |  arrays as VTKArrays
     |  
     |  Method resolution order:
     |      DataObject
     |      VTKObjectWrapper
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  GetAttributes(self, type)
     |      Returns the attributes specified by the type as a DataSetAttributes
     |      instance.
     |  
     |  GetFieldData(self)
     |      Returns the field data as a DataSetAttributes instance.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  FieldData
     |      This property returns the field data of a data object.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from VTKObjectWrapper:
     |  
     |  __getattr__(self, name)
     |      Forwards unknown attribute requests to VTK object.
     |  
     |  __init__(self, vtkobject)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from VTKObjectWrapper:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class DataSet(DataObject)
     |  DataSet(vtkobject)
     |  
     |  This is a python friendly wrapper of a vtkDataSet that defines
     |  a few useful properties.
     |  
     |  Method resolution order:
     |      DataSet
     |      DataObject
     |      VTKObjectWrapper
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  GetCellData(self)
     |      Returns the cell data as a DataSetAttributes instance.
     |  
     |  GetPointData(self)
     |      Returns the point data as a DataSetAttributes instance.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  CellData
     |      This property returns the cell data of a dataset.
     |  
     |  PointData
     |      This property returns the point data of the dataset.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from DataObject:
     |  
     |  GetAttributes(self, type)
     |      Returns the attributes specified by the type as a DataSetAttributes
     |      instance.
     |  
     |  GetFieldData(self)
     |      Returns the field data as a DataSetAttributes instance.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from DataObject:
     |  
     |  FieldData
     |      This property returns the field data of a data object.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from VTKObjectWrapper:
     |  
     |  __getattr__(self, name)
     |      Forwards unknown attribute requests to VTK object.
     |  
     |  __init__(self, vtkobject)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from VTKObjectWrapper:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class DataSetAttributes(VTKObjectWrapper)
     |  DataSetAttributes(vtkobject, dataset, association)
     |  
     |  This is a python friendly wrapper of vtkDataSetAttributes. It
     |  returns VTKArrays. It also provides the dictionary interface.
     |  
     |  Method resolution order:
     |      DataSetAttributes
     |      VTKObjectWrapper
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  GetArray(self, idx)
     |      Given an index or name, returns a VTKArray.
     |  
     |  PassData(self, other)
     |      A wrapper for vtkDataSet.PassData.
     |  
     |  __getitem__(self, idx)
     |      Implements the [] operator. Accepts an array name or index.
     |  
     |  __init__(self, vtkobject, dataset, association)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  append(self, narray, name)
     |      Appends a new array to the dataset attributes.
     |  
     |  keys(self)
     |      Returns the names of the arrays as a list.
     |  
     |  values(self)
     |      Returns the arrays as a list.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from VTKObjectWrapper:
     |  
     |  __getattr__(self, name)
     |      Forwards unknown attribute requests to VTK object.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from VTKObjectWrapper:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class Graph(DataObject)
     |  Graph(vtkobject)
     |  
     |  This is a python friendly wrapper of a vtkGraph that defines
     |  a few useful properties.
     |  
     |  Method resolution order:
     |      Graph
     |      DataObject
     |      VTKObjectWrapper
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  GetEdgeData(self)
     |      Returns the edge data as a DataSetAttributes instance.
     |  
     |  GetVertexData(self)
     |      Returns the vertex data as a DataSetAttributes instance.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  EdgeData
     |      This property returns the edge data of the graph.
     |  
     |  VertexData
     |      This property returns the vertex data of the graph.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from DataObject:
     |  
     |  GetAttributes(self, type)
     |      Returns the attributes specified by the type as a DataSetAttributes
     |      instance.
     |  
     |  GetFieldData(self)
     |      Returns the field data as a DataSetAttributes instance.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from DataObject:
     |  
     |  FieldData
     |      This property returns the field data of a data object.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from VTKObjectWrapper:
     |  
     |  __getattr__(self, name)
     |      Forwards unknown attribute requests to VTK object.
     |  
     |  __init__(self, vtkobject)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from VTKObjectWrapper:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class Molecule(DataObject)
     |  Molecule(vtkobject)
     |  
     |  This is a python friendly wrapper of a vtkMolecule that defines
     |  a few useful properties.
     |  
     |  Method resolution order:
     |      Molecule
     |      DataObject
     |      VTKObjectWrapper
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  GetAtomData(self)
     |      Returns the atom data as a DataSetAttributes instance.
     |  
     |  GetBondData(self)
     |      Returns the bond data as a DataSetAttributes instance.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  AtomData
     |      This property returns the atom data of the molecule.
     |  
     |  BondData
     |      This property returns the bond data of the molecule.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from DataObject:
     |  
     |  GetAttributes(self, type)
     |      Returns the attributes specified by the type as a DataSetAttributes
     |      instance.
     |  
     |  GetFieldData(self)
     |      Returns the field data as a DataSetAttributes instance.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from DataObject:
     |  
     |  FieldData
     |      This property returns the field data of a data object.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from VTKObjectWrapper:
     |  
     |  __getattr__(self, name)
     |      Forwards unknown attribute requests to VTK object.
     |  
     |  __init__(self, vtkobject)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from VTKObjectWrapper:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class MultiCompositeDataIterator(CompositeDataIterator)
     |  MultiCompositeDataIterator(cds)
     |  
     |  Iterator that can be used to iterate over multiple
     |  composite datasets together. This iterator works only
     |  with arrays that were copied from an original using
     |  CopyStructured. The most common use case is to use
     |  CopyStructure, then iterate over input and output together
     |  while creating output datasets from corresponding input
     |  datasets.
     |  
     |  Method resolution order:
     |      MultiCompositeDataIterator
     |      CompositeDataIterator
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, cds)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __next__(self)
     |  
     |  next(self)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from CompositeDataIterator:
     |  
     |  __getattr__(self, name)
     |      Returns attributes from the vtkCompositeDataIterator.
     |  
     |  __iter__(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from CompositeDataIterator:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class PointSet(DataSet)
     |  PointSet(vtkobject)
     |  
     |  This is a python friendly wrapper of a vtkPointSet that defines
     |  a few useful properties.
     |  
     |  Method resolution order:
     |      PointSet
     |      DataSet
     |      DataObject
     |      VTKObjectWrapper
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  GetPoints(self)
     |      Returns the points as a VTKArray instance. Returns None if the
     |      dataset has implicit points.
     |  
     |  SetPoints(self, pts)
     |      Given a VTKArray instance, sets the points of the dataset.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  Points
     |      This property returns the point coordinates of dataset.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from DataSet:
     |  
     |  GetCellData(self)
     |      Returns the cell data as a DataSetAttributes instance.
     |  
     |  GetPointData(self)
     |      Returns the point data as a DataSetAttributes instance.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from DataSet:
     |  
     |  CellData
     |      This property returns the cell data of a dataset.
     |  
     |  PointData
     |      This property returns the point data of the dataset.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from DataObject:
     |  
     |  GetAttributes(self, type)
     |      Returns the attributes specified by the type as a DataSetAttributes
     |      instance.
     |  
     |  GetFieldData(self)
     |      Returns the field data as a DataSetAttributes instance.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from DataObject:
     |  
     |  FieldData
     |      This property returns the field data of a data object.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from VTKObjectWrapper:
     |  
     |  __getattr__(self, name)
     |      Forwards unknown attribute requests to VTK object.
     |  
     |  __init__(self, vtkobject)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from VTKObjectWrapper:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class PolyData(PointSet)
     |  PolyData(vtkobject)
     |  
     |  This is a python friendly wrapper of a vtkPolyData that defines
     |  a few useful properties.
     |  
     |  Method resolution order:
     |      PolyData
     |      PointSet
     |      DataSet
     |      DataObject
     |      VTKObjectWrapper
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  GetPolygons(self)
     |      Returns the polys as a VTKArray instance.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  Polygons
     |      This property returns the connectivity of polygons.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from PointSet:
     |  
     |  GetPoints(self)
     |      Returns the points as a VTKArray instance. Returns None if the
     |      dataset has implicit points.
     |  
     |  SetPoints(self, pts)
     |      Given a VTKArray instance, sets the points of the dataset.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from PointSet:
     |  
     |  Points
     |      This property returns the point coordinates of dataset.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from DataSet:
     |  
     |  GetCellData(self)
     |      Returns the cell data as a DataSetAttributes instance.
     |  
     |  GetPointData(self)
     |      Returns the point data as a DataSetAttributes instance.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from DataSet:
     |  
     |  CellData
     |      This property returns the cell data of a dataset.
     |  
     |  PointData
     |      This property returns the point data of the dataset.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from DataObject:
     |  
     |  GetAttributes(self, type)
     |      Returns the attributes specified by the type as a DataSetAttributes
     |      instance.
     |  
     |  GetFieldData(self)
     |      Returns the field data as a DataSetAttributes instance.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from DataObject:
     |  
     |  FieldData
     |      This property returns the field data of a data object.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from VTKObjectWrapper:
     |  
     |  __getattr__(self, name)
     |      Forwards unknown attribute requests to VTK object.
     |  
     |  __init__(self, vtkobject)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from VTKObjectWrapper:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class Table(DataObject)
     |  Table(vtkobject)
     |  
     |  A wrapper for vtkFielData that makes it easier to access RowData array as
     |  VTKArrays
     |  
     |  Method resolution order:
     |      Table
     |      DataObject
     |      VTKObjectWrapper
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  GetRowData(self)
     |      Returns the row data as a DataSetAttributes instance.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  RowData
     |      This property returns the row data of the table.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from DataObject:
     |  
     |  GetAttributes(self, type)
     |      Returns the attributes specified by the type as a DataSetAttributes
     |      instance.
     |  
     |  GetFieldData(self)
     |      Returns the field data as a DataSetAttributes instance.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from DataObject:
     |  
     |  FieldData
     |      This property returns the field data of a data object.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from VTKObjectWrapper:
     |  
     |  __getattr__(self, name)
     |      Forwards unknown attribute requests to VTK object.
     |  
     |  __init__(self, vtkobject)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from VTKObjectWrapper:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class UnstructuredGrid(PointSet)
     |  UnstructuredGrid(vtkobject)
     |  
     |  This is a python friendly wrapper of a vtkUnstructuredGrid that defines
     |  a few useful properties.
     |  
     |  Method resolution order:
     |      UnstructuredGrid
     |      PointSet
     |      DataSet
     |      DataObject
     |      VTKObjectWrapper
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  GetCellLocations(self)
     |      Returns the cell locations as a VTKArray instance.
     |  
     |  GetCellTypes(self)
     |      Returns the cell types as a VTKArray instance.
     |  
     |  GetCells(self)
     |      Returns the cells as a VTKArray instance.
     |  
     |  SetCells(self, cellTypes, cellLocations, cells)
     |      Given cellTypes, cellLocations, cells as VTKArrays,
     |      populates the unstructured grid data structures.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  CellLocations
     |      This property returns the locations of cells.
     |  
     |  CellTypes
     |      This property returns the types of cells.
     |  
     |  Cells
     |      This property returns the connectivity of cells.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from PointSet:
     |  
     |  GetPoints(self)
     |      Returns the points as a VTKArray instance. Returns None if the
     |      dataset has implicit points.
     |  
     |  SetPoints(self, pts)
     |      Given a VTKArray instance, sets the points of the dataset.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from PointSet:
     |  
     |  Points
     |      This property returns the point coordinates of dataset.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from DataSet:
     |  
     |  GetCellData(self)
     |      Returns the cell data as a DataSetAttributes instance.
     |  
     |  GetPointData(self)
     |      Returns the point data as a DataSetAttributes instance.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from DataSet:
     |  
     |  CellData
     |      This property returns the cell data of a dataset.
     |  
     |  PointData
     |      This property returns the point data of the dataset.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from DataObject:
     |  
     |  GetAttributes(self, type)
     |      Returns the attributes specified by the type as a DataSetAttributes
     |      instance.
     |  
     |  GetFieldData(self)
     |      Returns the field data as a DataSetAttributes instance.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from DataObject:
     |  
     |  FieldData
     |      This property returns the field data of a data object.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from VTKObjectWrapper:
     |  
     |  __getattr__(self, name)
     |      Forwards unknown attribute requests to VTK object.
     |  
     |  __init__(self, vtkobject)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from VTKObjectWrapper:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class VTKArray(numpy.ndarray)
     |  VTKArray(input_array, array=None, dataset=None)
     |  
     |  This is a sub-class of numpy ndarray that stores a
     |  reference to a vtk array as well as the owning dataset.
     |  The numpy array and vtk array should point to the same
     |  memory location.
     |  
     |  Method resolution order:
     |      VTKArray
     |      numpy.ndarray
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __add__(self, other)
     |  
     |  __array_finalize__(self, obj)
     |      None.
     |  
     |  __array_wrap__(self, out_arr, context=None)
     |      a.__array_wrap__(obj) -> Object of same type as ndarray object a.
     |  
     |  __eq__(self, other)
     |  
     |  __floordiv__(self, other)
     |  
     |  __ge__(self, other)
     |  
     |  __getattr__(self, name)
     |      Forwards unknown attribute requests to VTK array.
     |  
     |  __gt__(self, other)
     |  
     |  __le__(self, other)
     |  
     |  __lshift__(self, other)
     |  
     |  __lt__(self, other)
     |  
     |  __mod__(self, other)
     |  
     |  __mul__(self, other)
     |  
     |  __ne__(self, other)
     |  
     |  __pow__(self, other)
     |  
     |  __radd__(self, other)
     |  
     |  __rfloordiv__(self, other)
     |  
     |  __rlshift__(self, other)
     |  
     |  __rmod__(self, other)
     |  
     |  __rmul__(self, other)
     |  
     |  __rpow__(self, other)
     |  
     |  __rrshift__(self, other)
     |  
     |  __rshift__(self, other)
     |  
     |  __rsub__(self, other)
     |  
     |  __rtruediv__(self, other)
     |  
     |  __rxor__(self, other)
     |  
     |  __sub__(self, other)
     |  
     |  __truediv__(self, other)
     |  
     |  __xor__(self, other)
     |  
     |  and(self, other)
     |  
     |  or(self, other)
     |  
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |  
     |  __new__(cls, input_array, array=None, dataset=None)
     |      Create and return a new object.  See help(type) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  DataSet
     |      Get the dataset this array is associated with. The reference to the
     |      dataset is held through a vtkWeakReference to ensure it doesn't prevent
     |      the dataset from being collected if necessary.
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  __hash__ = None
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from numpy.ndarray:
     |  
     |  __abs__(self, /)
     |      abs(self)
     |  
     |  __and__(self, value, /)
     |      Return self&value.
     |  
     |  __array__(...)
     |      a.__array__([dtype], /) -> reference if type unchanged, copy otherwise.
     |      
     |      Returns either a new reference to self if dtype is not given or a new array
     |      of provided data type if dtype is different from the current dtype of the
     |      array.
     |  
     |  __array_function__(...)
     |  
     |  __array_prepare__(...)
     |      a.__array_prepare__(obj) -> Object of same type as ndarray object obj.
     |  
     |  __array_ufunc__(...)
     |  
     |  __bool__(self, /)
     |      self != 0
     |  
     |  __complex__(...)
     |  
     |  __contains__(self, key, /)
     |      Return key in self.
     |  
     |  __copy__(...)
     |      a.__copy__()
     |      
     |      Used if :func:`copy.copy` is called on an array. Returns a copy of the array.
     |      
     |      Equivalent to ``a.copy(order='K')``.
     |  
     |  __deepcopy__(...)
     |      a.__deepcopy__(memo, /) -> Deep copy of array.
     |      
     |      Used if :func:`copy.deepcopy` is called on an array.
     |  
     |  __delitem__(self, key, /)
     |      Delete self[key].
     |  
     |  __divmod__(self, value, /)
     |      Return divmod(self, value).
     |  
     |  __float__(self, /)
     |      float(self)
     |  
     |  __format__(...)
     |      Default object formatter.
     |  
     |  __getitem__(self, key, /)
     |      Return self[key].
     |  
     |  __iadd__(self, value, /)
     |      Return self+=value.
     |  
     |  __iand__(self, value, /)
     |      Return self&=value.
     |  
     |  __ifloordiv__(self, value, /)
     |      Return self//=value.
     |  
     |  __ilshift__(self, value, /)
     |      Return self<<=value.
     |  
     |  __imatmul__(self, value, /)
     |      Return self@=value.
     |  
     |  __imod__(self, value, /)
     |      Return self%=value.
     |  
     |  __imul__(self, value, /)
     |      Return self*=value.
     |  
     |  __index__(self, /)
     |      Return self converted to an integer, if self is suitable for use as an index into a list.
     |  
     |  __int__(self, /)
     |      int(self)
     |  
     |  __invert__(self, /)
     |      ~self
     |  
     |  __ior__(self, value, /)
     |      Return self|=value.
     |  
     |  __ipow__(self, value, /)
     |      Return self**=value.
     |  
     |  __irshift__(self, value, /)
     |      Return self>>=value.
     |  
     |  __isub__(self, value, /)
     |      Return self-=value.
     |  
     |  __iter__(self, /)
     |      Implement iter(self).
     |  
     |  __itruediv__(self, value, /)
     |      Return self/=value.
     |  
     |  __ixor__(self, value, /)
     |      Return self^=value.
     |  
     |  __len__(self, /)
     |      Return len(self).
     |  
     |  __matmul__(self, value, /)
     |      Return self@value.
     |  
     |  __neg__(self, /)
     |      -self
     |  
     |  __or__(self, value, /)
     |      Return self|value.
     |  
     |  __pos__(self, /)
     |      +self
     |  
     |  __rand__(self, value, /)
     |      Return value&self.
     |  
     |  __rdivmod__(self, value, /)
     |      Return divmod(value, self).
     |  
     |  __reduce__(...)
     |      a.__reduce__()
     |      
     |      For pickling.
     |  
     |  __reduce_ex__(...)
     |      Helper for pickle.
     |  
     |  __repr__(self, /)
     |      Return repr(self).
     |  
     |  __rmatmul__(self, value, /)
     |      Return value@self.
     |  
     |  __ror__(self, value, /)
     |      Return value|self.
     |  
     |  __setitem__(self, key, value, /)
     |      Set self[key] to value.
     |  
     |  __setstate__(...)
     |      a.__setstate__(state, /)
     |      
     |      For unpickling.
     |      
     |      The `state` argument must be a sequence that contains the following
     |      elements:
     |      
     |      Parameters
     |      ----------
     |      version : int
     |          optional pickle version. If omitted defaults to 0.
     |      shape : tuple
     |      dtype : data-type
     |      isFortran : bool
     |      rawdata : string or list
     |          a binary string with the data (or a list if 'a' is an object array)
     |  
     |  __sizeof__(...)
     |      Size of object in memory, in bytes.
     |  
     |  __str__(self, /)
     |      Return str(self).
     |  
     |  all(...)
     |      a.all(axis=None, out=None, keepdims=False)
     |      
     |      Returns True if all elements evaluate to True.
     |      
     |      Refer to `numpy.all` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.all : equivalent function
     |  
     |  any(...)
     |      a.any(axis=None, out=None, keepdims=False)
     |      
     |      Returns True if any of the elements of `a` evaluate to True.
     |      
     |      Refer to `numpy.any` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.any : equivalent function
     |  
     |  argmax(...)
     |      a.argmax(axis=None, out=None)
     |      
     |      Return indices of the maximum values along the given axis.
     |      
     |      Refer to `numpy.argmax` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.argmax : equivalent function
     |  
     |  argmin(...)
     |      a.argmin(axis=None, out=None)
     |      
     |      Return indices of the minimum values along the given axis of `a`.
     |      
     |      Refer to `numpy.argmin` for detailed documentation.
     |      
     |      See Also
     |      --------
     |      numpy.argmin : equivalent function
     |  
     |  argpartition(...)
     |      a.argpartition(kth, axis=-1, kind='introselect', order=None)
     |      
     |      Returns the indices that would partition this array.
     |      
     |      Refer to `numpy.argpartition` for full documentation.
     |      
     |      .. versionadded:: 1.8.0
     |      
     |      See Also
     |      --------
     |      numpy.argpartition : equivalent function
     |  
     |  argsort(...)
     |      a.argsort(axis=-1, kind=None, order=None)
     |      
     |      Returns the indices that would sort this array.
     |      
     |      Refer to `numpy.argsort` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.argsort : equivalent function
     |  
     |  astype(...)
     |      a.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)
     |      
     |      Copy of the array, cast to a specified type.
     |      
     |      Parameters
     |      ----------
     |      dtype : str or dtype
     |          Typecode or data-type to which the array is cast.
     |      order : {'C', 'F', 'A', 'K'}, optional
     |          Controls the memory layout order of the result.
     |          'C' means C order, 'F' means Fortran order, 'A'
     |          means 'F' order if all the arrays are Fortran contiguous,
     |          'C' order otherwise, and 'K' means as close to the
     |          order the array elements appear in memory as possible.
     |          Default is 'K'.
     |      casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
     |          Controls what kind of data casting may occur. Defaults to 'unsafe'
     |          for backwards compatibility.
     |      
     |            * 'no' means the data types should not be cast at all.
     |            * 'equiv' means only byte-order changes are allowed.
     |            * 'safe' means only casts which can preserve values are allowed.
     |            * 'same_kind' means only safe casts or casts within a kind,
     |              like float64 to float32, are allowed.
     |            * 'unsafe' means any data conversions may be done.
     |      subok : bool, optional
     |          If True, then sub-classes will be passed-through (default), otherwise
     |          the returned array will be forced to be a base-class array.
     |      copy : bool, optional
     |          By default, astype always returns a newly allocated array. If this
     |          is set to false, and the `dtype`, `order`, and `subok`
     |          requirements are satisfied, the input array is returned instead
     |          of a copy.
     |      
     |      Returns
     |      -------
     |      arr_t : ndarray
     |          Unless `copy` is False and the other conditions for returning the input
     |          array are satisfied (see description for `copy` input parameter), `arr_t`
     |          is a new array of the same shape as the input array, with dtype, order
     |          given by `dtype`, `order`.
     |      
     |      Notes
     |      -----
     |      .. versionchanged:: 1.17.0
     |         Casting between a simple data type and a structured one is possible only
     |         for "unsafe" casting.  Casting to multiple fields is allowed, but
     |         casting from multiple fields is not.
     |      
     |      .. versionchanged:: 1.9.0
     |         Casting from numeric to string types in 'safe' casting mode requires
     |         that the string dtype length is long enough to store the max
     |         integer/float value converted.
     |      
     |      Raises
     |      ------
     |      ComplexWarning
     |          When casting from complex to float or int. To avoid this,
     |          one should use ``a.real.astype(t)``.
     |      
     |      Examples
     |      --------
     |      >>> x = np.array([1, 2, 2.5])
     |      >>> x
     |      array([1. ,  2. ,  2.5])
     |      
     |      >>> x.astype(int)
     |      array([1, 2, 2])
     |  
     |  byteswap(...)
     |      a.byteswap(inplace=False)
     |      
     |      Swap the bytes of the array elements
     |      
     |      Toggle between low-endian and big-endian data representation by
     |      returning a byteswapped array, optionally swapped in-place.
     |      Arrays of byte-strings are not swapped. The real and imaginary
     |      parts of a complex number are swapped individually.
     |      
     |      Parameters
     |      ----------
     |      inplace : bool, optional
     |          If ``True``, swap bytes in-place, default is ``False``.
     |      
     |      Returns
     |      -------
     |      out : ndarray
     |          The byteswapped array. If `inplace` is ``True``, this is
     |          a view to self.
     |      
     |      Examples
     |      --------
     |      >>> A = np.array([1, 256, 8755], dtype=np.int16)
     |      >>> list(map(hex, A))
     |      ['0x1', '0x100', '0x2233']
     |      >>> A.byteswap(inplace=True)
     |      array([  256,     1, 13090], dtype=int16)
     |      >>> list(map(hex, A))
     |      ['0x100', '0x1', '0x3322']
     |      
     |      Arrays of byte-strings are not swapped
     |      
     |      >>> A = np.array([b'ceg', b'fac'])
     |      >>> A.byteswap()
     |      array([b'ceg', b'fac'], dtype='|S3')
     |      
     |      ``A.newbyteorder().byteswap()`` produces an array with the same values
     |        but different representation in memory
     |      
     |      >>> A = np.array([1, 2, 3])
     |      >>> A.view(np.uint8)
     |      array([1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0,
     |             0, 0], dtype=uint8)
     |      >>> A.newbyteorder().byteswap(inplace=True)
     |      array([1, 2, 3])
     |      >>> A.view(np.uint8)
     |      array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
     |             0, 3], dtype=uint8)
     |  
     |  choose(...)
     |      a.choose(choices, out=None, mode='raise')
     |      
     |      Use an index array to construct a new array from a set of choices.
     |      
     |      Refer to `numpy.choose` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.choose : equivalent function
     |  
     |  clip(...)
     |      a.clip(min=None, max=None, out=None, **kwargs)
     |      
     |      Return an array whose values are limited to ``[min, max]``.
     |      One of max or min must be given.
     |      
     |      Refer to `numpy.clip` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.clip : equivalent function
     |  
     |  compress(...)
     |      a.compress(condition, axis=None, out=None)
     |      
     |      Return selected slices of this array along given axis.
     |      
     |      Refer to `numpy.compress` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.compress : equivalent function
     |  
     |  conj(...)
     |      a.conj()
     |      
     |      Complex-conjugate all elements.
     |      
     |      Refer to `numpy.conjugate` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.conjugate : equivalent function
     |  
     |  conjugate(...)
     |      a.conjugate()
     |      
     |      Return the complex conjugate, element-wise.
     |      
     |      Refer to `numpy.conjugate` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.conjugate : equivalent function
     |  
     |  copy(...)
     |      a.copy(order='C')
     |      
     |      Return a copy of the array.
     |      
     |      Parameters
     |      ----------
     |      order : {'C', 'F', 'A', 'K'}, optional
     |          Controls the memory layout of the copy. 'C' means C-order,
     |          'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
     |          'C' otherwise. 'K' means match the layout of `a` as closely
     |          as possible. (Note that this function and :func:`numpy.copy` are very
     |          similar, but have different default values for their order=
     |          arguments.)
     |      
     |      See also
     |      --------
     |      numpy.copy
     |      numpy.copyto
     |      
     |      Examples
     |      --------
     |      >>> x = np.array([[1,2,3],[4,5,6]], order='F')
     |      
     |      >>> y = x.copy()
     |      
     |      >>> x.fill(0)
     |      
     |      >>> x
     |      array([[0, 0, 0],
     |             [0, 0, 0]])
     |      
     |      >>> y
     |      array([[1, 2, 3],
     |             [4, 5, 6]])
     |      
     |      >>> y.flags['C_CONTIGUOUS']
     |      True
     |  
     |  cumprod(...)
     |      a.cumprod(axis=None, dtype=None, out=None)
     |      
     |      Return the cumulative product of the elements along the given axis.
     |      
     |      Refer to `numpy.cumprod` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.cumprod : equivalent function
     |  
     |  cumsum(...)
     |      a.cumsum(axis=None, dtype=None, out=None)
     |      
     |      Return the cumulative sum of the elements along the given axis.
     |      
     |      Refer to `numpy.cumsum` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.cumsum : equivalent function
     |  
     |  diagonal(...)
     |      a.diagonal(offset=0, axis1=0, axis2=1)
     |      
     |      Return specified diagonals. In NumPy 1.9 the returned array is a
     |      read-only view instead of a copy as in previous NumPy versions.  In
     |      a future version the read-only restriction will be removed.
     |      
     |      Refer to :func:`numpy.diagonal` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.diagonal : equivalent function
     |  
     |  dot(...)
     |      a.dot(b, out=None)
     |      
     |      Dot product of two arrays.
     |      
     |      Refer to `numpy.dot` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.dot : equivalent function
     |      
     |      Examples
     |      --------
     |      >>> a = np.eye(2)
     |      >>> b = np.ones((2, 2)) * 2
     |      >>> a.dot(b)
     |      array([[2.,  2.],
     |             [2.,  2.]])
     |      
     |      This array method can be conveniently chained:
     |      
     |      >>> a.dot(b).dot(b)
     |      array([[8.,  8.],
     |             [8.,  8.]])
     |  
     |  dump(...)
     |      a.dump(file)
     |      
     |      Dump a pickle of the array to the specified file.
     |      The array can be read back with pickle.load or numpy.load.
     |      
     |      Parameters
     |      ----------
     |      file : str or Path
     |          A string naming the dump file.
     |      
     |          .. versionchanged:: 1.17.0
     |              `pathlib.Path` objects are now accepted.
     |  
     |  dumps(...)
     |      a.dumps()
     |      
     |      Returns the pickle of the array as a string.
     |      pickle.loads or numpy.loads will convert the string back to an array.
     |      
     |      Parameters
     |      ----------
     |      None
     |  
     |  fill(...)
     |      a.fill(value)
     |      
     |      Fill the array with a scalar value.
     |      
     |      Parameters
     |      ----------
     |      value : scalar
     |          All elements of `a` will be assigned this value.
     |      
     |      Examples
     |      --------
     |      >>> a = np.array([1, 2])
     |      >>> a.fill(0)
     |      >>> a
     |      array([0, 0])
     |      >>> a = np.empty(2)
     |      >>> a.fill(1)
     |      >>> a
     |      array([1.,  1.])
     |  
     |  flatten(...)
     |      a.flatten(order='C')
     |      
     |      Return a copy of the array collapsed into one dimension.
     |      
     |      Parameters
     |      ----------
     |      order : {'C', 'F', 'A', 'K'}, optional
     |          'C' means to flatten in row-major (C-style) order.
     |          'F' means to flatten in column-major (Fortran-
     |          style) order. 'A' means to flatten in column-major
     |          order if `a` is Fortran *contiguous* in memory,
     |          row-major order otherwise. 'K' means to flatten
     |          `a` in the order the elements occur in memory.
     |          The default is 'C'.
     |      
     |      Returns
     |      -------
     |      y : ndarray
     |          A copy of the input array, flattened to one dimension.
     |      
     |      See Also
     |      --------
     |      ravel : Return a flattened array.
     |      flat : A 1-D flat iterator over the array.
     |      
     |      Examples
     |      --------
     |      >>> a = np.array([[1,2], [3,4]])
     |      >>> a.flatten()
     |      array([1, 2, 3, 4])
     |      >>> a.flatten('F')
     |      array([1, 3, 2, 4])
     |  
     |  getfield(...)
     |      a.getfield(dtype, offset=0)
     |      
     |      Returns a field of the given array as a certain type.
     |      
     |      A field is a view of the array data with a given data-type. The values in
     |      the view are determined by the given type and the offset into the current
     |      array in bytes. The offset needs to be such that the view dtype fits in the
     |      array dtype; for example an array of dtype complex128 has 16-byte elements.
     |      If taking a view with a 32-bit integer (4 bytes), the offset needs to be
     |      between 0 and 12 bytes.
     |      
     |      Parameters
     |      ----------
     |      dtype : str or dtype
     |          The data type of the view. The dtype size of the view can not be larger
     |          than that of the array itself.
     |      offset : int
     |          Number of bytes to skip before beginning the element view.
     |      
     |      Examples
     |      --------
     |      >>> x = np.diag([1.+1.j]*2)
     |      >>> x[1, 1] = 2 + 4.j
     |      >>> x
     |      array([[1.+1.j,  0.+0.j],
     |             [0.+0.j,  2.+4.j]])
     |      >>> x.getfield(np.float64)
     |      array([[1.,  0.],
     |             [0.,  2.]])
     |      
     |      By choosing an offset of 8 bytes we can select the complex part of the
     |      array for our view:
     |      
     |      >>> x.getfield(np.float64, offset=8)
     |      array([[1.,  0.],
     |             [0.,  4.]])
     |  
     |  item(...)
     |      a.item(*args)
     |      
     |      Copy an element of an array to a standard Python scalar and return it.
     |      
     |      Parameters
     |      ----------
     |      \*args : Arguments (variable number and type)
     |      
     |          * none: in this case, the method only works for arrays
     |            with one element (`a.size == 1`), which element is
     |            copied into a standard Python scalar object and returned.
     |      
     |          * int_type: this argument is interpreted as a flat index into
     |            the array, specifying which element to copy and return.
     |      
     |          * tuple of int_types: functions as does a single int_type argument,
     |            except that the argument is interpreted as an nd-index into the
     |            array.
     |      
     |      Returns
     |      -------
     |      z : Standard Python scalar object
     |          A copy of the specified element of the array as a suitable
     |          Python scalar
     |      
     |      Notes
     |      -----
     |      When the data type of `a` is longdouble or clongdouble, item() returns
     |      a scalar array object because there is no available Python scalar that
     |      would not lose information. Void arrays return a buffer object for item(),
     |      unless fields are defined, in which case a tuple is returned.
     |      
     |      `item` is very similar to a[args], except, instead of an array scalar,
     |      a standard Python scalar is returned. This can be useful for speeding up
     |      access to elements of the array and doing arithmetic on elements of the
     |      array using Python's optimized math.
     |      
     |      Examples
     |      --------
     |      >>> np.random.seed(123)
     |      >>> x = np.random.randint(9, size=(3, 3))
     |      >>> x
     |      array([[2, 2, 6],
     |             [1, 3, 6],
     |             [1, 0, 1]])
     |      >>> x.item(3)
     |      1
     |      >>> x.item(7)
     |      0
     |      >>> x.item((0, 1))
     |      2
     |      >>> x.item((2, 2))
     |      1
     |  
     |  itemset(...)
     |      a.itemset(*args)
     |      
     |      Insert scalar into an array (scalar is cast to array's dtype, if possible)
     |      
     |      There must be at least 1 argument, and define the last argument
     |      as *item*.  Then, ``a.itemset(*args)`` is equivalent to but faster
     |      than ``a[args] = item``.  The item should be a scalar value and `args`
     |      must select a single item in the array `a`.
     |      
     |      Parameters
     |      ----------
     |      \*args : Arguments
     |          If one argument: a scalar, only used in case `a` is of size 1.
     |          If two arguments: the last argument is the value to be set
     |          and must be a scalar, the first argument specifies a single array
     |          element location. It is either an int or a tuple.
     |      
     |      Notes
     |      -----
     |      Compared to indexing syntax, `itemset` provides some speed increase
     |      for placing a scalar into a particular location in an `ndarray`,
     |      if you must do this.  However, generally this is discouraged:
     |      among other problems, it complicates the appearance of the code.
     |      Also, when using `itemset` (and `item`) inside a loop, be sure
     |      to assign the methods to a local variable to avoid the attribute
     |      look-up at each loop iteration.
     |      
     |      Examples
     |      --------
     |      >>> np.random.seed(123)
     |      >>> x = np.random.randint(9, size=(3, 3))
     |      >>> x
     |      array([[2, 2, 6],
     |             [1, 3, 6],
     |             [1, 0, 1]])
     |      >>> x.itemset(4, 0)
     |      >>> x.itemset((2, 2), 9)
     |      >>> x
     |      array([[2, 2, 6],
     |             [1, 0, 6],
     |             [1, 0, 9]])
     |  
     |  max(...)
     |      a.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)
     |      
     |      Return the maximum along a given axis.
     |      
     |      Refer to `numpy.amax` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.amax : equivalent function
     |  
     |  mean(...)
     |      a.mean(axis=None, dtype=None, out=None, keepdims=False)
     |      
     |      Returns the average of the array elements along given axis.
     |      
     |      Refer to `numpy.mean` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.mean : equivalent function
     |  
     |  min(...)
     |      a.min(axis=None, out=None, keepdims=False, initial=<no value>, where=True)
     |      
     |      Return the minimum along a given axis.
     |      
     |      Refer to `numpy.amin` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.amin : equivalent function
     |  
     |  newbyteorder(...)
     |      arr.newbyteorder(new_order='S')
     |      
     |      Return the array with the same data viewed with a different byte order.
     |      
     |      Equivalent to::
     |      
     |          arr.view(arr.dtype.newbytorder(new_order))
     |      
     |      Changes are also made in all fields and sub-arrays of the array data
     |      type.
     |      
     |      
     |      
     |      Parameters
     |      ----------
     |      new_order : string, optional
     |          Byte order to force; a value from the byte order specifications
     |          below. `new_order` codes can be any of:
     |      
     |          * 'S' - swap dtype from current to opposite endian
     |          * {'<', 'L'} - little endian
     |          * {'>', 'B'} - big endian
     |          * {'=', 'N'} - native order
     |          * {'|', 'I'} - ignore (no change to byte order)
     |      
     |          The default value ('S') results in swapping the current
     |          byte order. The code does a case-insensitive check on the first
     |          letter of `new_order` for the alternatives above.  For example,
     |          any of 'B' or 'b' or 'biggish' are valid to specify big-endian.
     |      
     |      
     |      Returns
     |      -------
     |      new_arr : array
     |          New array object with the dtype reflecting given change to the
     |          byte order.
     |  
     |  nonzero(...)
     |      a.nonzero()
     |      
     |      Return the indices of the elements that are non-zero.
     |      
     |      Refer to `numpy.nonzero` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.nonzero : equivalent function
     |  
     |  partition(...)
     |      a.partition(kth, axis=-1, kind='introselect', order=None)
     |      
     |      Rearranges the elements in the array in such a way that the value of the
     |      element in kth position is in the position it would be in a sorted array.
     |      All elements smaller than the kth element are moved before this element and
     |      all equal or greater are moved behind it. The ordering of the elements in
     |      the two partitions is undefined.
     |      
     |      .. versionadded:: 1.8.0
     |      
     |      Parameters
     |      ----------
     |      kth : int or sequence of ints
     |          Element index to partition by. The kth element value will be in its
     |          final sorted position and all smaller elements will be moved before it
     |          and all equal or greater elements behind it.
     |          The order of all elements in the partitions is undefined.
     |          If provided with a sequence of kth it will partition all elements
     |          indexed by kth of them into their sorted position at once.
     |      axis : int, optional
     |          Axis along which to sort. Default is -1, which means sort along the
     |          last axis.
     |      kind : {'introselect'}, optional
     |          Selection algorithm. Default is 'introselect'.
     |      order : str or list of str, optional
     |          When `a` is an array with fields defined, this argument specifies
     |          which fields to compare first, second, etc. A single field can
     |          be specified as a string, and not all fields need to be specified,
     |          but unspecified fields will still be used, in the order in which
     |          they come up in the dtype, to break ties.
     |      
     |      See Also
     |      --------
     |      numpy.partition : Return a parititioned copy of an array.
     |      argpartition : Indirect partition.
     |      sort : Full sort.
     |      
     |      Notes
     |      -----
     |      See ``np.partition`` for notes on the different algorithms.
     |      
     |      Examples
     |      --------
     |      >>> a = np.array([3, 4, 2, 1])
     |      >>> a.partition(3)
     |      >>> a
     |      array([2, 1, 3, 4])
     |      
     |      >>> a.partition((1, 3))
     |      >>> a
     |      array([1, 2, 3, 4])
     |  
     |  prod(...)
     |      a.prod(axis=None, dtype=None, out=None, keepdims=False, initial=1, where=True)
     |      
     |      Return the product of the array elements over the given axis
     |      
     |      Refer to `numpy.prod` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.prod : equivalent function
     |  
     |  ptp(...)
     |      a.ptp(axis=None, out=None, keepdims=False)
     |      
     |      Peak to peak (maximum - minimum) value along a given axis.
     |      
     |      Refer to `numpy.ptp` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.ptp : equivalent function
     |  
     |  put(...)
     |      a.put(indices, values, mode='raise')
     |      
     |      Set ``a.flat[n] = values[n]`` for all `n` in indices.
     |      
     |      Refer to `numpy.put` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.put : equivalent function
     |  
     |  ravel(...)
     |      a.ravel([order])
     |      
     |      Return a flattened array.
     |      
     |      Refer to `numpy.ravel` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.ravel : equivalent function
     |      
     |      ndarray.flat : a flat iterator on the array.
     |  
     |  repeat(...)
     |      a.repeat(repeats, axis=None)
     |      
     |      Repeat elements of an array.
     |      
     |      Refer to `numpy.repeat` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.repeat : equivalent function
     |  
     |  reshape(...)
     |      a.reshape(shape, order='C')
     |      
     |      Returns an array containing the same data with a new shape.
     |      
     |      Refer to `numpy.reshape` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.reshape : equivalent function
     |      
     |      Notes
     |      -----
     |      Unlike the free function `numpy.reshape`, this method on `ndarray` allows
     |      the elements of the shape parameter to be passed in as separate arguments.
     |      For example, ``a.reshape(10, 11)`` is equivalent to
     |      ``a.reshape((10, 11))``.
     |  
     |  resize(...)
     |      a.resize(new_shape, refcheck=True)
     |      
     |      Change shape and size of array in-place.
     |      
     |      Parameters
     |      ----------
     |      new_shape : tuple of ints, or `n` ints
     |          Shape of resized array.
     |      refcheck : bool, optional
     |          If False, reference count will not be checked. Default is True.
     |      
     |      Returns
     |      -------
     |      None
     |      
     |      Raises
     |      ------
     |      ValueError
     |          If `a` does not own its own data or references or views to it exist,
     |          and the data memory must be changed.
     |          PyPy only: will always raise if the data memory must be changed, since
     |          there is no reliable way to determine if references or views to it
     |          exist.
     |      
     |      SystemError
     |          If the `order` keyword argument is specified. This behaviour is a
     |          bug in NumPy.
     |      
     |      See Also
     |      --------
     |      resize : Return a new array with the specified shape.
     |      
     |      Notes
     |      -----
     |      This reallocates space for the data area if necessary.
     |      
     |      Only contiguous arrays (data elements consecutive in memory) can be
     |      resized.
     |      
     |      The purpose of the reference count check is to make sure you
     |      do not use this array as a buffer for another Python object and then
     |      reallocate the memory. However, reference counts can increase in
     |      other ways so if you are sure that you have not shared the memory
     |      for this array with another Python object, then you may safely set
     |      `refcheck` to False.
     |      
     |      Examples
     |      --------
     |      Shrinking an array: array is flattened (in the order that the data are
     |      stored in memory), resized, and reshaped:
     |      
     |      >>> a = np.array([[0, 1], [2, 3]], order='C')
     |      >>> a.resize((2, 1))
     |      >>> a
     |      array([[0],
     |             [1]])
     |      
     |      >>> a = np.array([[0, 1], [2, 3]], order='F')
     |      >>> a.resize((2, 1))
     |      >>> a
     |      array([[0],
     |             [2]])
     |      
     |      Enlarging an array: as above, but missing entries are filled with zeros:
     |      
     |      >>> b = np.array([[0, 1], [2, 3]])
     |      >>> b.resize(2, 3) # new_shape parameter doesn't have to be a tuple
     |      >>> b
     |      array([[0, 1, 2],
     |             [3, 0, 0]])
     |      
     |      Referencing an array prevents resizing...
     |      
     |      >>> c = a
     |      >>> a.resize((1, 1))
     |      Traceback (most recent call last):
     |      ...
     |      ValueError: cannot resize an array that references or is referenced ...
     |      
     |      Unless `refcheck` is False:
     |      
     |      >>> a.resize((1, 1), refcheck=False)
     |      >>> a
     |      array([[0]])
     |      >>> c
     |      array([[0]])
     |  
     |  round(...)
     |      a.round(decimals=0, out=None)
     |      
     |      Return `a` with each element rounded to the given number of decimals.
     |      
     |      Refer to `numpy.around` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.around : equivalent function
     |  
     |  searchsorted(...)
     |      a.searchsorted(v, side='left', sorter=None)
     |      
     |      Find indices where elements of v should be inserted in a to maintain order.
     |      
     |      For full documentation, see `numpy.searchsorted`
     |      
     |      See Also
     |      --------
     |      numpy.searchsorted : equivalent function
     |  
     |  setfield(...)
     |      a.setfield(val, dtype, offset=0)
     |      
     |      Put a value into a specified place in a field defined by a data-type.
     |      
     |      Place `val` into `a`'s field defined by `dtype` and beginning `offset`
     |      bytes into the field.
     |      
     |      Parameters
     |      ----------
     |      val : object
     |          Value to be placed in field.
     |      dtype : dtype object
     |          Data-type of the field in which to place `val`.
     |      offset : int, optional
     |          The number of bytes into the field at which to place `val`.
     |      
     |      Returns
     |      -------
     |      None
     |      
     |      See Also
     |      --------
     |      getfield
     |      
     |      Examples
     |      --------
     |      >>> x = np.eye(3)
     |      >>> x.getfield(np.float64)
     |      array([[1.,  0.,  0.],
     |             [0.,  1.,  0.],
     |             [0.,  0.,  1.]])
     |      >>> x.setfield(3, np.int32)
     |      >>> x.getfield(np.int32)
     |      array([[3, 3, 3],
     |             [3, 3, 3],
     |             [3, 3, 3]], dtype=int32)
     |      >>> x
     |      array([[1.0e+000, 1.5e-323, 1.5e-323],
     |             [1.5e-323, 1.0e+000, 1.5e-323],
     |             [1.5e-323, 1.5e-323, 1.0e+000]])
     |      >>> x.setfield(np.eye(3), np.int32)
     |      >>> x
     |      array([[1.,  0.,  0.],
     |             [0.,  1.,  0.],
     |             [0.,  0.,  1.]])
     |  
     |  setflags(...)
     |      a.setflags(write=None, align=None, uic=None)
     |      
     |      Set array flags WRITEABLE, ALIGNED, (WRITEBACKIFCOPY and UPDATEIFCOPY),
     |      respectively.
     |      
     |      These Boolean-valued flags affect how numpy interprets the memory
     |      area used by `a` (see Notes below). The ALIGNED flag can only
     |      be set to True if the data is actually aligned according to the type.
     |      The WRITEBACKIFCOPY and (deprecated) UPDATEIFCOPY flags can never be set
     |      to True. The flag WRITEABLE can only be set to True if the array owns its
     |      own memory, or the ultimate owner of the memory exposes a writeable buffer
     |      interface, or is a string. (The exception for string is made so that
     |      unpickling can be done without copying memory.)
     |      
     |      Parameters
     |      ----------
     |      write : bool, optional
     |          Describes whether or not `a` can be written to.
     |      align : bool, optional
     |          Describes whether or not `a` is aligned properly for its type.
     |      uic : bool, optional
     |          Describes whether or not `a` is a copy of another "base" array.
     |      
     |      Notes
     |      -----
     |      Array flags provide information about how the memory area used
     |      for the array is to be interpreted. There are 7 Boolean flags
     |      in use, only four of which can be changed by the user:
     |      WRITEBACKIFCOPY, UPDATEIFCOPY, WRITEABLE, and ALIGNED.
     |      
     |      WRITEABLE (W) the data area can be written to;
     |      
     |      ALIGNED (A) the data and strides are aligned appropriately for the hardware
     |      (as determined by the compiler);
     |      
     |      UPDATEIFCOPY (U) (deprecated), replaced by WRITEBACKIFCOPY;
     |      
     |      WRITEBACKIFCOPY (X) this array is a copy of some other array (referenced
     |      by .base). When the C-API function PyArray_ResolveWritebackIfCopy is
     |      called, the base array will be updated with the contents of this array.
     |      
     |      All flags can be accessed using the single (upper case) letter as well
     |      as the full name.
     |      
     |      Examples
     |      --------
     |      >>> y = np.array([[3, 1, 7],
     |      ...               [2, 0, 0],
     |      ...               [8, 5, 9]])
     |      >>> y
     |      array([[3, 1, 7],
     |             [2, 0, 0],
     |             [8, 5, 9]])
     |      >>> y.flags
     |        C_CONTIGUOUS : True
     |        F_CONTIGUOUS : False
     |        OWNDATA : True
     |        WRITEABLE : True
     |        ALIGNED : True
     |        WRITEBACKIFCOPY : False
     |        UPDATEIFCOPY : False
     |      >>> y.setflags(write=0, align=0)
     |      >>> y.flags
     |        C_CONTIGUOUS : True
     |        F_CONTIGUOUS : False
     |        OWNDATA : True
     |        WRITEABLE : False
     |        ALIGNED : False
     |        WRITEBACKIFCOPY : False
     |        UPDATEIFCOPY : False
     |      >>> y.setflags(uic=1)
     |      Traceback (most recent call last):
     |        File "<stdin>", line 1, in <module>
     |      ValueError: cannot set WRITEBACKIFCOPY flag to True
     |  
     |  sort(...)
     |      a.sort(axis=-1, kind=None, order=None)
     |      
     |      Sort an array in-place. Refer to `numpy.sort` for full documentation.
     |      
     |      Parameters
     |      ----------
     |      axis : int, optional
     |          Axis along which to sort. Default is -1, which means sort along the
     |          last axis.
     |      kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
     |          Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
     |          and 'mergesort' use timsort under the covers and, in general, the
     |          actual implementation will vary with datatype. The 'mergesort' option
     |          is retained for backwards compatibility.
     |      
     |          .. versionchanged:: 1.15.0.
     |             The 'stable' option was added.
     |      
     |      order : str or list of str, optional
     |          When `a` is an array with fields defined, this argument specifies
     |          which fields to compare first, second, etc.  A single field can
     |          be specified as a string, and not all fields need be specified,
     |          but unspecified fields will still be used, in the order in which
     |          they come up in the dtype, to break ties.
     |      
     |      See Also
     |      --------
     |      numpy.sort : Return a sorted copy of an array.
     |      numpy.argsort : Indirect sort.
     |      numpy.lexsort : Indirect stable sort on multiple keys.
     |      numpy.searchsorted : Find elements in sorted array.
     |      numpy.partition: Partial sort.
     |      
     |      Notes
     |      -----
     |      See `numpy.sort` for notes on the different sorting algorithms.
     |      
     |      Examples
     |      --------
     |      >>> a = np.array([[1,4], [3,1]])
     |      >>> a.sort(axis=1)
     |      >>> a
     |      array([[1, 4],
     |             [1, 3]])
     |      >>> a.sort(axis=0)
     |      >>> a
     |      array([[1, 3],
     |             [1, 4]])
     |      
     |      Use the `order` keyword to specify a field to use when sorting a
     |      structured array:
     |      
     |      >>> a = np.array([('a', 2), ('c', 1)], dtype=[('x', 'S1'), ('y', int)])
     |      >>> a.sort(order='y')
     |      >>> a
     |      array([(b'c', 1), (b'a', 2)],
     |            dtype=[('x', 'S1'), ('y', '<i8')])
     |  
     |  squeeze(...)
     |      a.squeeze(axis=None)
     |      
     |      Remove single-dimensional entries from the shape of `a`.
     |      
     |      Refer to `numpy.squeeze` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.squeeze : equivalent function
     |  
     |  std(...)
     |      a.std(axis=None, dtype=None, out=None, ddof=0, keepdims=False)
     |      
     |      Returns the standard deviation of the array elements along given axis.
     |      
     |      Refer to `numpy.std` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.std : equivalent function
     |  
     |  sum(...)
     |      a.sum(axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True)
     |      
     |      Return the sum of the array elements over the given axis.
     |      
     |      Refer to `numpy.sum` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.sum : equivalent function
     |  
     |  swapaxes(...)
     |      a.swapaxes(axis1, axis2)
     |      
     |      Return a view of the array with `axis1` and `axis2` interchanged.
     |      
     |      Refer to `numpy.swapaxes` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.swapaxes : equivalent function
     |  
     |  take(...)
     |      a.take(indices, axis=None, out=None, mode='raise')
     |      
     |      Return an array formed from the elements of `a` at the given indices.
     |      
     |      Refer to `numpy.take` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.take : equivalent function
     |  
     |  tobytes(...)
     |      a.tobytes(order='C')
     |      
     |      Construct Python bytes containing the raw data bytes in the array.
     |      
     |      Constructs Python bytes showing a copy of the raw contents of
     |      data memory. The bytes object can be produced in either 'C' or 'Fortran',
     |      or 'Any' order (the default is 'C'-order). 'Any' order means C-order
     |      unless the F_CONTIGUOUS flag in the array is set, in which case it
     |      means 'Fortran' order.
     |      
     |      .. versionadded:: 1.9.0
     |      
     |      Parameters
     |      ----------
     |      order : {'C', 'F', None}, optional
     |          Order of the data for multidimensional arrays:
     |          C, Fortran, or the same as for the original array.
     |      
     |      Returns
     |      -------
     |      s : bytes
     |          Python bytes exhibiting a copy of `a`'s raw data.
     |      
     |      Examples
     |      --------
     |      >>> x = np.array([[0, 1], [2, 3]], dtype='<u2')
     |      >>> x.tobytes()
     |      b'\x00\x00\x01\x00\x02\x00\x03\x00'
     |      >>> x.tobytes('C') == x.tobytes()
     |      True
     |      >>> x.tobytes('F')
     |      b'\x00\x00\x02\x00\x01\x00\x03\x00'
     |  
     |  tofile(...)
     |      a.tofile(fid, sep="", format="%s")
     |      
     |      Write array to a file as text or binary (default).
     |      
     |      Data is always written in 'C' order, independent of the order of `a`.
     |      The data produced by this method can be recovered using the function
     |      fromfile().
     |      
     |      Parameters
     |      ----------
     |      fid : file or str or Path
     |          An open file object, or a string containing a filename.
     |      
     |          .. versionchanged:: 1.17.0
     |              `pathlib.Path` objects are now accepted.
     |      
     |      sep : str
     |          Separator between array items for text output.
     |          If "" (empty), a binary file is written, equivalent to
     |          ``file.write(a.tobytes())``.
     |      format : str
     |          Format string for text file output.
     |          Each entry in the array is formatted to text by first converting
     |          it to the closest Python type, and then using "format" % item.
     |      
     |      Notes
     |      -----
     |      This is a convenience function for quick storage of array data.
     |      Information on endianness and precision is lost, so this method is not a
     |      good choice for files intended to archive data or transport data between
     |      machines with different endianness. Some of these problems can be overcome
     |      by outputting the data as text files, at the expense of speed and file
     |      size.
     |      
     |      When fid is a file object, array contents are directly written to the
     |      file, bypassing the file object's ``write`` method. As a result, tofile
     |      cannot be used with files objects supporting compression (e.g., GzipFile)
     |      or file-like objects that do not support ``fileno()`` (e.g., BytesIO).
     |  
     |  tolist(...)
     |      a.tolist()
     |      
     |      Return the array as an ``a.ndim``-levels deep nested list of Python scalars.
     |      
     |      Return a copy of the array data as a (nested) Python list.
     |      Data items are converted to the nearest compatible builtin Python type, via
     |      the `~numpy.ndarray.item` function.
     |      
     |      If ``a.ndim`` is 0, then since the depth of the nested list is 0, it will
     |      not be a list at all, but a simple Python scalar.
     |      
     |      Parameters
     |      ----------
     |      none
     |      
     |      Returns
     |      -------
     |      y : object, or list of object, or list of list of object, or ...
     |          The possibly nested list of array elements.
     |      
     |      Notes
     |      -----
     |      The array may be recreated via ``a = np.array(a.tolist())``, although this
     |      may sometimes lose precision.
     |      
     |      Examples
     |      --------
     |      For a 1D array, ``a.tolist()`` is almost the same as ``list(a)``,
     |      except that ``tolist`` changes numpy scalars to Python scalars:
     |      
     |      >>> a = np.uint32([1, 2])
     |      >>> a_list = list(a)
     |      >>> a_list
     |      [1, 2]
     |      >>> type(a_list[0])
     |      <class 'numpy.uint32'>
     |      >>> a_tolist = a.tolist()
     |      >>> a_tolist
     |      [1, 2]
     |      >>> type(a_tolist[0])
     |      <class 'int'>
     |      
     |      Additionally, for a 2D array, ``tolist`` applies recursively:
     |      
     |      >>> a = np.array([[1, 2], [3, 4]])
     |      >>> list(a)
     |      [array([1, 2]), array([3, 4])]
     |      >>> a.tolist()
     |      [[1, 2], [3, 4]]
     |      
     |      The base case for this recursion is a 0D array:
     |      
     |      >>> a = np.array(1)
     |      >>> list(a)
     |      Traceback (most recent call last):
     |        ...
     |      TypeError: iteration over a 0-d array
     |      >>> a.tolist()
     |      1
     |  
     |  tostring(...)
     |      a.tostring(order='C')
     |      
     |      A compatibility alias for `tobytes`, with exactly the same behavior.
     |      
     |      Despite its name, it returns `bytes` not `str`\ s.
     |      
     |      .. deprecated:: 1.19.0
     |  
     |  trace(...)
     |      a.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)
     |      
     |      Return the sum along diagonals of the array.
     |      
     |      Refer to `numpy.trace` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.trace : equivalent function
     |  
     |  transpose(...)
     |      a.transpose(*axes)
     |      
     |      Returns a view of the array with axes transposed.
     |      
     |      For a 1-D array this has no effect, as a transposed vector is simply the
     |      same vector. To convert a 1-D array into a 2D column vector, an additional
     |      dimension must be added. `np.atleast2d(a).T` achieves this, as does
     |      `a[:, np.newaxis]`.
     |      For a 2-D array, this is a standard matrix transpose.
     |      For an n-D array, if axes are given, their order indicates how the
     |      axes are permuted (see Examples). If axes are not provided and
     |      ``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
     |      ``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.
     |      
     |      Parameters
     |      ----------
     |      axes : None, tuple of ints, or `n` ints
     |      
     |       * None or no argument: reverses the order of the axes.
     |      
     |       * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
     |         `i`-th axis becomes `a.transpose()`'s `j`-th axis.
     |      
     |       * `n` ints: same as an n-tuple of the same ints (this form is
     |         intended simply as a "convenience" alternative to the tuple form)
     |      
     |      Returns
     |      -------
     |      out : ndarray
     |          View of `a`, with axes suitably permuted.
     |      
     |      See Also
     |      --------
     |      ndarray.T : Array property returning the array transposed.
     |      ndarray.reshape : Give a new shape to an array without changing its data.
     |      
     |      Examples
     |      --------
     |      >>> a = np.array([[1, 2], [3, 4]])
     |      >>> a
     |      array([[1, 2],
     |             [3, 4]])
     |      >>> a.transpose()
     |      array([[1, 3],
     |             [2, 4]])
     |      >>> a.transpose((1, 0))
     |      array([[1, 3],
     |             [2, 4]])
     |      >>> a.transpose(1, 0)
     |      array([[1, 3],
     |             [2, 4]])
     |  
     |  var(...)
     |      a.var(axis=None, dtype=None, out=None, ddof=0, keepdims=False)
     |      
     |      Returns the variance of the array elements, along given axis.
     |      
     |      Refer to `numpy.var` for full documentation.
     |      
     |      See Also
     |      --------
     |      numpy.var : equivalent function
     |  
     |  view(...)
     |      a.view([dtype][, type])
     |      
     |      New view of array with the same data.
     |      
     |      .. note::
     |          Passing None for ``dtype`` is different from omitting the parameter,
     |          since the former invokes ``dtype(None)`` which is an alias for
     |          ``dtype('float_')``.
     |      
     |      Parameters
     |      ----------
     |      dtype : data-type or ndarray sub-class, optional
     |          Data-type descriptor of the returned view, e.g., float32 or int16.
     |          Omitting it results in the view having the same data-type as `a`.
     |          This argument can also be specified as an ndarray sub-class, which
     |          then specifies the type of the returned object (this is equivalent to
     |          setting the ``type`` parameter).
     |      type : Python type, optional
     |          Type of the returned view, e.g., ndarray or matrix.  Again, omission
     |          of the parameter results in type preservation.
     |      
     |      Notes
     |      -----
     |      ``a.view()`` is used two different ways:
     |      
     |      ``a.view(some_dtype)`` or ``a.view(dtype=some_dtype)`` constructs a view
     |      of the array's memory with a different data-type.  This can cause a
     |      reinterpretation of the bytes of memory.
     |      
     |      ``a.view(ndarray_subclass)`` or ``a.view(type=ndarray_subclass)`` just
     |      returns an instance of `ndarray_subclass` that looks at the same array
     |      (same shape, dtype, etc.)  This does not cause a reinterpretation of the
     |      memory.
     |      
     |      For ``a.view(some_dtype)``, if ``some_dtype`` has a different number of
     |      bytes per entry than the previous dtype (for example, converting a
     |      regular array to a structured array), then the behavior of the view
     |      cannot be predicted just from the superficial appearance of ``a`` (shown
     |      by ``print(a)``). It also depends on exactly how ``a`` is stored in
     |      memory. Therefore if ``a`` is C-ordered versus fortran-ordered, versus
     |      defined as a slice or transpose, etc., the view may give different
     |      results.
     |      
     |      
     |      Examples
     |      --------
     |      >>> x = np.array([(1, 2)], dtype=[('a', np.int8), ('b', np.int8)])
     |      
     |      Viewing array data using a different type and dtype:
     |      
     |      >>> y = x.view(dtype=np.int16, type=np.matrix)
     |      >>> y
     |      matrix([[513]], dtype=int16)
     |      >>> print(type(y))
     |      <class 'numpy.matrix'>
     |      
     |      Creating a view on a structured array so it can be used in calculations
     |      
     |      >>> x = np.array([(1, 2),(3,4)], dtype=[('a', np.int8), ('b', np.int8)])
     |      >>> xv = x.view(dtype=np.int8).reshape(-1,2)
     |      >>> xv
     |      array([[1, 2],
     |             [3, 4]], dtype=int8)
     |      >>> xv.mean(0)
     |      array([2.,  3.])
     |      
     |      Making changes to the view changes the underlying array
     |      
     |      >>> xv[0,1] = 20
     |      >>> x
     |      array([(1, 20), (3,  4)], dtype=[('a', 'i1'), ('b', 'i1')])
     |      
     |      Using a view to convert an array to a recarray:
     |      
     |      >>> z = x.view(np.recarray)
     |      >>> z.a
     |      array([1, 3], dtype=int8)
     |      
     |      Views share data:
     |      
     |      >>> x[0] = (9, 10)
     |      >>> z[0]
     |      (9, 10)
     |      
     |      Views that change the dtype size (bytes per entry) should normally be
     |      avoided on arrays defined by slices, transposes, fortran-ordering, etc.:
     |      
     |      >>> x = np.array([[1,2,3],[4,5,6]], dtype=np.int16)
     |      >>> y = x[:, 0:2]
     |      >>> y
     |      array([[1, 2],
     |             [4, 5]], dtype=int16)
     |      >>> y.view(dtype=[('width', np.int16), ('length', np.int16)])
     |      Traceback (most recent call last):
     |          ...
     |      ValueError: To change to a dtype of a different size, the array must be C-contiguous
     |      >>> z = y.copy()
     |      >>> z.view(dtype=[('width', np.int16), ('length', np.int16)])
     |      array([[(1, 2)],
     |             [(4, 5)]], dtype=[('width', '<i2'), ('length', '<i2')])
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from numpy.ndarray:
     |  
     |  T
     |      The transposed array.
     |      
     |      Same as ``self.transpose()``.
     |      
     |      Examples
     |      --------
     |      >>> x = np.array([[1.,2.],[3.,4.]])
     |      >>> x
     |      array([[ 1.,  2.],
     |             [ 3.,  4.]])
     |      >>> x.T
     |      array([[ 1.,  3.],
     |             [ 2.,  4.]])
     |      >>> x = np.array([1.,2.,3.,4.])
     |      >>> x
     |      array([ 1.,  2.,  3.,  4.])
     |      >>> x.T
     |      array([ 1.,  2.,  3.,  4.])
     |      
     |      See Also
     |      --------
     |      transpose
     |  
     |  __array_interface__
     |      Array protocol: Python side.
     |  
     |  __array_priority__
     |      Array priority.
     |  
     |  __array_struct__
     |      Array protocol: C-struct side.
     |  
     |  base
     |      Base object if memory is from some other object.
     |      
     |      Examples
     |      --------
     |      The base of an array that owns its memory is None:
     |      
     |      >>> x = np.array([1,2,3,4])
     |      >>> x.base is None
     |      True
     |      
     |      Slicing creates a view, whose memory is shared with x:
     |      
     |      >>> y = x[2:]
     |      >>> y.base is x
     |      True
     |  
     |  ctypes
     |      An object to simplify the interaction of the array with the ctypes
     |      module.
     |      
     |      This attribute creates an object that makes it easier to use arrays
     |      when calling shared libraries with the ctypes module. The returned
     |      object has, among others, data, shape, and strides attributes (see
     |      Notes below) which themselves return ctypes objects that can be used
     |      as arguments to a shared library.
     |      
     |      Parameters
     |      ----------
     |      None
     |      
     |      Returns
     |      -------
     |      c : Python object
     |          Possessing attributes data, shape, strides, etc.
     |      
     |      See Also
     |      --------
     |      numpy.ctypeslib
     |      
     |      Notes
     |      -----
     |      Below are the public attributes of this object which were documented
     |      in "Guide to NumPy" (we have omitted undocumented public attributes,
     |      as well as documented private attributes):
     |      
     |      .. autoattribute:: numpy.core._internal._ctypes.data
     |          :noindex:
     |      
     |      .. autoattribute:: numpy.core._internal._ctypes.shape
     |          :noindex:
     |      
     |      .. autoattribute:: numpy.core._internal._ctypes.strides
     |          :noindex:
     |      
     |      .. automethod:: numpy.core._internal._ctypes.data_as
     |          :noindex:
     |      
     |      .. automethod:: numpy.core._internal._ctypes.shape_as
     |          :noindex:
     |      
     |      .. automethod:: numpy.core._internal._ctypes.strides_as
     |          :noindex:
     |      
     |      If the ctypes module is not available, then the ctypes attribute
     |      of array objects still returns something useful, but ctypes objects
     |      are not returned and errors may be raised instead. In particular,
     |      the object will still have the ``as_parameter`` attribute which will
     |      return an integer equal to the data attribute.
     |      
     |      Examples
     |      --------
     |      >>> import ctypes
     |      >>> x = np.array([[0, 1], [2, 3]], dtype=np.int32)
     |      >>> x
     |      array([[0, 1],
     |             [2, 3]], dtype=int32)
     |      >>> x.ctypes.data
     |      31962608 # may vary
     |      >>> x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
     |      <__main__.LP_c_uint object at 0x7ff2fc1fc200> # may vary
     |      >>> x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)).contents
     |      c_uint(0)
     |      >>> x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)).contents
     |      c_ulong(4294967296)
     |      >>> x.ctypes.shape
     |      <numpy.core._internal.c_long_Array_2 object at 0x7ff2fc1fce60> # may vary
     |      >>> x.ctypes.strides
     |      <numpy.core._internal.c_long_Array_2 object at 0x7ff2fc1ff320> # may vary
     |  
     |  data
     |      Python buffer object pointing to the start of the array's data.
     |  
     |  dtype
     |      Data-type of the array's elements.
     |      
     |      Parameters
     |      ----------
     |      None
     |      
     |      Returns
     |      -------
     |      d : numpy dtype object
     |      
     |      See Also
     |      --------
     |      numpy.dtype
     |      
     |      Examples
     |      --------
     |      >>> x
     |      array([[0, 1],
     |             [2, 3]])
     |      >>> x.dtype
     |      dtype('int32')
     |      >>> type(x.dtype)
     |      <type 'numpy.dtype'>
     |  
     |  flags
     |      Information about the memory layout of the array.
     |      
     |      Attributes
     |      ----------
     |      C_CONTIGUOUS (C)
     |          The data is in a single, C-style contiguous segment.
     |      F_CONTIGUOUS (F)
     |          The data is in a single, Fortran-style contiguous segment.
     |      OWNDATA (O)
     |          The array owns the memory it uses or borrows it from another object.
     |      WRITEABLE (W)
     |          The data area can be written to.  Setting this to False locks
     |          the data, making it read-only.  A view (slice, etc.) inherits WRITEABLE
     |          from its base array at creation time, but a view of a writeable
     |          array may be subsequently locked while the base array remains writeable.
     |          (The opposite is not true, in that a view of a locked array may not
     |          be made writeable.  However, currently, locking a base object does not
     |          lock any views that already reference it, so under that circumstance it
     |          is possible to alter the contents of a locked array via a previously
     |          created writeable view onto it.)  Attempting to change a non-writeable
     |          array raises a RuntimeError exception.
     |      ALIGNED (A)
     |          The data and all elements are aligned appropriately for the hardware.
     |      WRITEBACKIFCOPY (X)
     |          This array is a copy of some other array. The C-API function
     |          PyArray_ResolveWritebackIfCopy must be called before deallocating
     |          to the base array will be updated with the contents of this array.
     |      UPDATEIFCOPY (U)
     |          (Deprecated, use WRITEBACKIFCOPY) This array is a copy of some other array.
     |          When this array is
     |          deallocated, the base array will be updated with the contents of
     |          this array.
     |      FNC
     |          F_CONTIGUOUS and not C_CONTIGUOUS.
     |      FORC
     |          F_CONTIGUOUS or C_CONTIGUOUS (one-segment test).
     |      BEHAVED (B)
     |          ALIGNED and WRITEABLE.
     |      CARRAY (CA)
     |          BEHAVED and C_CONTIGUOUS.
     |      FARRAY (FA)
     |          BEHAVED and F_CONTIGUOUS and not C_CONTIGUOUS.
     |      
     |      Notes
     |      -----
     |      The `flags` object can be accessed dictionary-like (as in ``a.flags['WRITEABLE']``),
     |      or by using lowercased attribute names (as in ``a.flags.writeable``). Short flag
     |      names are only supported in dictionary access.
     |      
     |      Only the WRITEBACKIFCOPY, UPDATEIFCOPY, WRITEABLE, and ALIGNED flags can be
     |      changed by the user, via direct assignment to the attribute or dictionary
     |      entry, or by calling `ndarray.setflags`.
     |      
     |      The array flags cannot be set arbitrarily:
     |      
     |      - UPDATEIFCOPY can only be set ``False``.
     |      - WRITEBACKIFCOPY can only be set ``False``.
     |      - ALIGNED can only be set ``True`` if the data is truly aligned.
     |      - WRITEABLE can only be set ``True`` if the array owns its own memory
     |        or the ultimate owner of the memory exposes a writeable buffer
     |        interface or is a string.
     |      
     |      Arrays can be both C-style and Fortran-style contiguous simultaneously.
     |      This is clear for 1-dimensional arrays, but can also be true for higher
     |      dimensional arrays.
     |      
     |      Even for contiguous arrays a stride for a given dimension
     |      ``arr.strides[dim]`` may be *arbitrary* if ``arr.shape[dim] == 1``
     |      or the array has no elements.
     |      It does *not* generally hold that ``self.strides[-1] == self.itemsize``
     |      for C-style contiguous arrays or ``self.strides[0] == self.itemsize`` for
     |      Fortran-style contiguous arrays is true.
     |  
     |  flat
     |      A 1-D iterator over the array.
     |      
     |      This is a `numpy.flatiter` instance, which acts similarly to, but is not
     |      a subclass of, Python's built-in iterator object.
     |      
     |      See Also
     |      --------
     |      flatten : Return a copy of the array collapsed into one dimension.
     |      
     |      flatiter
     |      
     |      Examples
     |      --------
     |      >>> x = np.arange(1, 7).reshape(2, 3)
     |      >>> x
     |      array([[1, 2, 3],
     |             [4, 5, 6]])
     |      >>> x.flat[3]
     |      4
     |      >>> x.T
     |      array([[1, 4],
     |             [2, 5],
     |             [3, 6]])
     |      >>> x.T.flat[3]
     |      5
     |      >>> type(x.flat)
     |      <class 'numpy.flatiter'>
     |      
     |      An assignment example:
     |      
     |      >>> x.flat = 3; x
     |      array([[3, 3, 3],
     |             [3, 3, 3]])
     |      >>> x.flat[[1,4]] = 1; x
     |      array([[3, 1, 3],
     |             [3, 1, 3]])
     |  
     |  imag
     |      The imaginary part of the array.
     |      
     |      Examples
     |      --------
     |      >>> x = np.sqrt([1+0j, 0+1j])
     |      >>> x.imag
     |      array([ 0.        ,  0.70710678])
     |      >>> x.imag.dtype
     |      dtype('float64')
     |  
     |  itemsize
     |      Length of one array element in bytes.
     |      
     |      Examples
     |      --------
     |      >>> x = np.array([1,2,3], dtype=np.float64)
     |      >>> x.itemsize
     |      8
     |      >>> x = np.array([1,2,3], dtype=np.complex128)
     |      >>> x.itemsize
     |      16
     |  
     |  nbytes
     |      Total bytes consumed by the elements of the array.
     |      
     |      Notes
     |      -----
     |      Does not include memory consumed by non-element attributes of the
     |      array object.
     |      
     |      Examples
     |      --------
     |      >>> x = np.zeros((3,5,2), dtype=np.complex128)
     |      >>> x.nbytes
     |      480
     |      >>> np.prod(x.shape) * x.itemsize
     |      480
     |  
     |  ndim
     |      Number of array dimensions.
     |      
     |      Examples
     |      --------
     |      >>> x = np.array([1, 2, 3])
     |      >>> x.ndim
     |      1
     |      >>> y = np.zeros((2, 3, 4))
     |      >>> y.ndim
     |      3
     |  
     |  real
     |      The real part of the array.
     |      
     |      Examples
     |      --------
     |      >>> x = np.sqrt([1+0j, 0+1j])
     |      >>> x.real
     |      array([ 1.        ,  0.70710678])
     |      >>> x.real.dtype
     |      dtype('float64')
     |      
     |      See Also
     |      --------
     |      numpy.real : equivalent function
     |  
     |  shape
     |      Tuple of array dimensions.
     |      
     |      The shape property is usually used to get the current shape of an array,
     |      but may also be used to reshape the array in-place by assigning a tuple of
     |      array dimensions to it.  As with `numpy.reshape`, one of the new shape
     |      dimensions can be -1, in which case its value is inferred from the size of
     |      the array and the remaining dimensions. Reshaping an array in-place will
     |      fail if a copy is required.
     |      
     |      Examples
     |      --------
     |      >>> x = np.array([1, 2, 3, 4])
     |      >>> x.shape
     |      (4,)
     |      >>> y = np.zeros((2, 3, 4))
     |      >>> y.shape
     |      (2, 3, 4)
     |      >>> y.shape = (3, 8)
     |      >>> y
     |      array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     |             [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     |             [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
     |      >>> y.shape = (3, 6)
     |      Traceback (most recent call last):
     |        File "<stdin>", line 1, in <module>
     |      ValueError: total size of new array must be unchanged
     |      >>> np.zeros((4,2))[::2].shape = (-1,)
     |      Traceback (most recent call last):
     |        File "<stdin>", line 1, in <module>
     |      AttributeError: Incompatible shape for in-place modification. Use
     |      `.reshape()` to make a copy with the desired shape.
     |      
     |      See Also
     |      --------
     |      numpy.reshape : similar function
     |      ndarray.reshape : similar method
     |  
     |  size
     |      Number of elements in the array.
     |      
     |      Equal to ``np.prod(a.shape)``, i.e., the product of the array's
     |      dimensions.
     |      
     |      Notes
     |      -----
     |      `a.size` returns a standard arbitrary precision Python integer. This
     |      may not be the case with other methods of obtaining the same value
     |      (like the suggested ``np.prod(a.shape)``, which returns an instance
     |      of ``np.int_``), and may be relevant if the value is used further in
     |      calculations that may overflow a fixed size integer type.
     |      
     |      Examples
     |      --------
     |      >>> x = np.zeros((3, 5, 2), dtype=np.complex128)
     |      >>> x.size
     |      30
     |      >>> np.prod(x.shape)
     |      30
     |  
     |  strides
     |      Tuple of bytes to step in each dimension when traversing an array.
     |      
     |      The byte offset of element ``(i[0], i[1], ..., i[n])`` in an array `a`
     |      is::
     |      
     |          offset = sum(np.array(i) * a.strides)
     |      
     |      A more detailed explanation of strides can be found in the
     |      "ndarray.rst" file in the NumPy reference guide.
     |      
     |      Notes
     |      -----
     |      Imagine an array of 32-bit integers (each 4 bytes)::
     |      
     |        x = np.array([[0, 1, 2, 3, 4],
     |                      [5, 6, 7, 8, 9]], dtype=np.int32)
     |      
     |      This array is stored in memory as 40 bytes, one after the other
     |      (known as a contiguous block of memory).  The strides of an array tell
     |      us how many bytes we have to skip in memory to move to the next position
     |      along a certain axis.  For example, we have to skip 4 bytes (1 value) to
     |      move to the next column, but 20 bytes (5 values) to get to the same
     |      position in the next row.  As such, the strides for the array `x` will be
     |      ``(20, 4)``.
     |      
     |      See Also
     |      --------
     |      numpy.lib.stride_tricks.as_strided
     |      
     |      Examples
     |      --------
     |      >>> y = np.reshape(np.arange(2*3*4), (2,3,4))
     |      >>> y
     |      array([[[ 0,  1,  2,  3],
     |              [ 4,  5,  6,  7],
     |              [ 8,  9, 10, 11]],
     |             [[12, 13, 14, 15],
     |              [16, 17, 18, 19],
     |              [20, 21, 22, 23]]])
     |      >>> y.strides
     |      (48, 16, 4)
     |      >>> y[1,1,1]
     |      17
     |      >>> offset=sum(y.strides * np.array((1,1,1)))
     |      >>> offset/y.itemsize
     |      17
     |      
     |      >>> x = np.reshape(np.arange(5*6*7*8), (5,6,7,8)).transpose(2,3,1,0)
     |      >>> x.strides
     |      (32, 4, 224, 1344)
     |      >>> i = np.array([3,5,2,2])
     |      >>> offset = sum(i * x.strides)
     |      >>> x[3,5,2,2]
     |      813
     |      >>> offset / x.itemsize
     |      813
    
    class VTKArrayMetaClass(builtins.type)
     |  VTKArrayMetaClass(name, parent, attr)
     |  
     |  type(object_or_name, bases, dict)
     |  type(object) -> the object's type
     |  type(name, bases, dict) -> a new type
     |  
     |  Method resolution order:
     |      VTKArrayMetaClass
     |      builtins.type
     |      builtins.object
     |  
     |  Static methods defined here:
     |  
     |  __new__(mcs, name, parent, attr)
     |      We overwrite numerical/comparison operators because we might need
     |      to reshape one of the arrays to perform the operation without
     |      broadcast errors. For instance:
     |      
     |      An array G of shape (n,3) resulted from computing the
     |      gradient on a scalar array S of shape (n,) cannot be added together without
     |      reshaping.
     |      G + expand_dims(S,1) works,
     |      G + S gives an error:
     |      ValueError: operands could not be broadcast together with shapes (n,3) (n,)
     |      
     |      This metaclass overwrites operators such that it computes this
     |      reshape operation automatically by appending 1s to the
     |      dimensions of the array with fewer dimensions.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from builtins.type:
     |  
     |  __call__(self, /, *args, **kwargs)
     |      Call self as a function.
     |  
     |  __delattr__(self, name, /)
     |      Implement delattr(self, name).
     |  
     |  __dir__(self, /)
     |      Specialized __dir__ implementation for types.
     |  
     |  __getattribute__(self, name, /)
     |      Return getattr(self, name).
     |  
     |  __init__(self, /, *args, **kwargs)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __instancecheck__(self, instance, /)
     |      Check if an object is an instance.
     |  
     |  __repr__(self, /)
     |      Return repr(self).
     |  
     |  __setattr__(self, name, value, /)
     |      Implement setattr(self, name, value).
     |  
     |  __sizeof__(self, /)
     |      Return memory consumption of the type object.
     |  
     |  __subclasscheck__(self, subclass, /)
     |      Check if a class is a subclass.
     |  
     |  __subclasses__(self, /)
     |      Return a list of immediate subclasses.
     |  
     |  mro(self, /)
     |      Return a type's method resolution order.
     |  
     |  ----------------------------------------------------------------------
     |  Class methods inherited from builtins.type:
     |  
     |  __prepare__(...) from builtins.type
     |      __prepare__() -> dict
     |      used to create the namespace for the class statement
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from builtins.type:
     |  
     |  __abstractmethods__
     |  
     |  __dict__
     |  
     |  __text_signature__
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from builtins.type:
     |  
     |  __base__ = <class 'type'>
     |      type(object_or_name, bases, dict)
     |      type(object) -> the object's type
     |      type(name, bases, dict) -> a new type
     |  
     |  __bases__ = (<class 'type'>,)
     |  
     |  __basicsize__ = 864
     |  
     |  __dictoffset__ = 264
     |  
     |  __flags__ = 2148292097
     |  
     |  __itemsize__ = 40
     |  
     |  __mro__ = (<class 'paraview.vtk.numpy_interface.dataset_adapter.VTKArr...
     |  
     |  __weakrefoffset__ = 368
    
    class VTKCompositeDataArray(builtins.object)
     |  VTKCompositeDataArray(arrays=[], dataset=None, name=None, association=None)
     |  
     |  This class manages a set of arrays of the same name contained
     |  within a composite dataset. Its main purpose is to provide a
     |  Numpy-type interface to composite data arrays which are naturally
     |  nothing but a collection of vtkDataArrays. A VTKCompositeDataArray
     |  makes such a collection appear as a single Numpy array and support
     |  all array operations that this module and the associated algorithm
     |  module support. Note that this is not a subclass of a Numpy array
     |  and as such cannot be passed to native Numpy functions. Instead
     |  VTK modules should be used to process composite arrays.
     |  
     |  Methods defined here:
     |  
     |  GetArrays(self)
     |      Returns the internal container of VTKArrays. If necessary,
     |      this will populate the array list from a composite dataset.
     |  
     |  GetSize(self)
     |      Returns the number of elements in the array.
     |  
     |  __add__(self, other)
     |  
     |  __and__(self, other)
     |  
     |  __eq__(self, other)
     |  
     |  __floordiv__(self, other)
     |  
     |  __ge__(self, other)
     |  
     |  __getitem__(self, index)
     |      Overwritten to refer indexing to underlying VTKArrays.
     |      For the most part, this will behave like Numpy. Note
     |      that indexing is done per array - arrays are never treated
     |      as forming a bigger array. If the index is another composite
     |      array, a one-to-one mapping between arrays is assumed.
     |  
     |  __gt__(self, other)
     |  
     |  __init__(self, arrays=[], dataset=None, name=None, association=None)
     |      Construct a composite array given a container of
     |      arrays, a dataset, name and association. It is sufficient
     |      to define a container of arrays to define a composite array.
     |      It is also possible to initialize an array by defining
     |      the dataset, name and array association. In that case,
     |      the underlying arrays will be created lazily when they
     |      are needed. It is recommended to use the latter method
     |      when initializing from an existing composite dataset.
     |  
     |  __le__(self, other)
     |  
     |  __lshift__(self, other)
     |  
     |  __lt__(self, other)
     |  
     |  __mod__(self, other)
     |  
     |  __mul__(self, other)
     |  
     |  __ne__(self, other)
     |  
     |  __or__(self, other)
     |  
     |  __pow__(self, other)
     |  
     |  __radd__(self, other)
     |  
     |  __rand__(self, other)
     |  
     |  __rfloordiv__(self, other)
     |  
     |  __rlshift__(self, other)
     |  
     |  __rmod__(self, other)
     |  
     |  __rmul__(self, other)
     |  
     |  __ror__(self, other)
     |  
     |  __rpow__(self, other)
     |  
     |  __rrshift__(self, other)
     |  
     |  __rshift__(self, other)
     |  
     |  __rsub__(self, other)
     |  
     |  __rtruediv__(self, other)
     |  
     |  __rxor__(self, other)
     |  
     |  __str__(self)
     |      Return str(self).
     |  
     |  __sub__(self, other)
     |  
     |  __truediv__(self, other)
     |  
     |  __xor__(self, other)
     |  
     |  astype(self, dtype)
     |      Implements numpy array's as array method.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  Arrays
     |      Returns the internal container of VTKArrays. If necessary,
     |      this will populate the array list from a composite dataset.
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  size
     |      Returns the number of elements in the array.
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  __hash__ = None
    
    class VTKCompositeDataArrayMetaClass(builtins.type)
     |  VTKCompositeDataArrayMetaClass(name, parent, attr)
     |  
     |  type(object_or_name, bases, dict)
     |  type(object) -> the object's type
     |  type(name, bases, dict) -> a new type
     |  
     |  Method resolution order:
     |      VTKCompositeDataArrayMetaClass
     |      builtins.type
     |      builtins.object
     |  
     |  Static methods defined here:
     |  
     |  __new__(mcs, name, parent, attr)
     |      Simplify the implementation of the numeric/logical sequence API.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from builtins.type:
     |  
     |  __call__(self, /, *args, **kwargs)
     |      Call self as a function.
     |  
     |  __delattr__(self, name, /)
     |      Implement delattr(self, name).
     |  
     |  __dir__(self, /)
     |      Specialized __dir__ implementation for types.
     |  
     |  __getattribute__(self, name, /)
     |      Return getattr(self, name).
     |  
     |  __init__(self, /, *args, **kwargs)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __instancecheck__(self, instance, /)
     |      Check if an object is an instance.
     |  
     |  __repr__(self, /)
     |      Return repr(self).
     |  
     |  __setattr__(self, name, value, /)
     |      Implement setattr(self, name, value).
     |  
     |  __sizeof__(self, /)
     |      Return memory consumption of the type object.
     |  
     |  __subclasscheck__(self, subclass, /)
     |      Check if a class is a subclass.
     |  
     |  __subclasses__(self, /)
     |      Return a list of immediate subclasses.
     |  
     |  mro(self, /)
     |      Return a type's method resolution order.
     |  
     |  ----------------------------------------------------------------------
     |  Class methods inherited from builtins.type:
     |  
     |  __prepare__(...) from builtins.type
     |      __prepare__() -> dict
     |      used to create the namespace for the class statement
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from builtins.type:
     |  
     |  __abstractmethods__
     |  
     |  __dict__
     |  
     |  __text_signature__
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from builtins.type:
     |  
     |  __base__ = <class 'type'>
     |      type(object_or_name, bases, dict)
     |      type(object) -> the object's type
     |      type(name, bases, dict) -> a new type
     |  
     |  __bases__ = (<class 'type'>,)
     |  
     |  __basicsize__ = 864
     |  
     |  __dictoffset__ = 264
     |  
     |  __flags__ = 2148292097
     |  
     |  __itemsize__ = 40
     |  
     |  __mro__ = (<class 'paraview.vtk.numpy_interface.dataset_adapter.VTKCom...
     |  
     |  __weakrefoffset__ = 368
    
    class VTKNoneArray(builtins.object)
     |  VTKNoneArray is used to represent a "void" array. An instance
     |  of this class (NoneArray) is returned instead of None when an
     |  array that doesn't exist in a DataSetAttributes is requested.
     |  All operations on the NoneArray return NoneArray. The main reason
     |  for this is to support operations in parallel where one of the
     |  processes may be working on an empty dataset. In such cases,
     |  the process is still expected to evaluate a whole expression because
     |  some of the functions may perform bulk MPI communication. None
     |  cannot be used in these instances because it cannot properly override
     |  operators such as __add__, __sub__ etc. This is the main raison
     |  d'etre for VTKNoneArray.
     |  
     |  Methods defined here:
     |  
     |  __add__(self, other)
     |  
     |  __and__(self, other)
     |  
     |  __eq__(self, other)
     |  
     |  __floordiv__(self, other)
     |  
     |  __ge__(self, other)
     |  
     |  __getitem__(self, index)
     |  
     |  __gt__(self, other)
     |  
     |  __le__(self, other)
     |  
     |  __lshift__(self, other)
     |  
     |  __lt__(self, other)
     |  
     |  __mod__(self, other)
     |  
     |  __mul__(self, other)
     |  
     |  __ne__(self, other)
     |  
     |  __or__(self, other)
     |  
     |  __pow__(self, other)
     |  
     |  __radd__(self, other)
     |  
     |  __rand__(self, other)
     |  
     |  __rfloordiv__(self, other)
     |  
     |  __rlshift__(self, other)
     |  
     |  __rmod__(self, other)
     |  
     |  __rmul__(self, other)
     |  
     |  __ror__(self, other)
     |  
     |  __rpow__(self, other)
     |  
     |  __rrshift__(self, other)
     |  
     |  __rshift__(self, other)
     |  
     |  __rsub__(self, other)
     |  
     |  __rtruediv__(self, other)
     |  
     |  __rxor__(self, other)
     |  
     |  __sub__(self, other)
     |  
     |  __truediv__(self, other)
     |  
     |  __xor__(self, other)
     |  
     |  astype(self, dtype)
     |      Implements numpy array's astype method.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  __hash__ = None
    
    class VTKNoneArrayMetaClass(builtins.type)
     |  VTKNoneArrayMetaClass(name, parent, attr)
     |  
     |  type(object_or_name, bases, dict)
     |  type(object) -> the object's type
     |  type(name, bases, dict) -> a new type
     |  
     |  Method resolution order:
     |      VTKNoneArrayMetaClass
     |      builtins.type
     |      builtins.object
     |  
     |  Static methods defined here:
     |  
     |  __new__(mcs, name, parent, attr)
     |      Simplify the implementation of the numeric/logical sequence API.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from builtins.type:
     |  
     |  __call__(self, /, *args, **kwargs)
     |      Call self as a function.
     |  
     |  __delattr__(self, name, /)
     |      Implement delattr(self, name).
     |  
     |  __dir__(self, /)
     |      Specialized __dir__ implementation for types.
     |  
     |  __getattribute__(self, name, /)
     |      Return getattr(self, name).
     |  
     |  __init__(self, /, *args, **kwargs)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __instancecheck__(self, instance, /)
     |      Check if an object is an instance.
     |  
     |  __repr__(self, /)
     |      Return repr(self).
     |  
     |  __setattr__(self, name, value, /)
     |      Implement setattr(self, name, value).
     |  
     |  __sizeof__(self, /)
     |      Return memory consumption of the type object.
     |  
     |  __subclasscheck__(self, subclass, /)
     |      Check if a class is a subclass.
     |  
     |  __subclasses__(self, /)
     |      Return a list of immediate subclasses.
     |  
     |  mro(self, /)
     |      Return a type's method resolution order.
     |  
     |  ----------------------------------------------------------------------
     |  Class methods inherited from builtins.type:
     |  
     |  __prepare__(...) from builtins.type
     |      __prepare__() -> dict
     |      used to create the namespace for the class statement
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from builtins.type:
     |  
     |  __abstractmethods__
     |  
     |  __dict__
     |  
     |  __text_signature__
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from builtins.type:
     |  
     |  __base__ = <class 'type'>
     |      type(object_or_name, bases, dict)
     |      type(object) -> the object's type
     |      type(name, bases, dict) -> a new type
     |  
     |  __bases__ = (<class 'type'>,)
     |  
     |  __basicsize__ = 864
     |  
     |  __dictoffset__ = 264
     |  
     |  __flags__ = 2148292097
     |  
     |  __itemsize__ = 40
     |  
     |  __mro__ = (<class 'paraview.vtk.numpy_interface.dataset_adapter.VTKNon...
     |  
     |  __weakrefoffset__ = 368
    
    class VTKObjectWrapper(builtins.object)
     |  VTKObjectWrapper(vtkobject)
     |  
     |  Superclass for classes that wrap VTK objects with Python objects.
     |  This class holds a reference to the wrapped VTK object. It also
     |  forwards unresolved methods to the underlying object by overloading
     |  __get__attr.
     |  
     |  Methods defined here:
     |  
     |  __getattr__(self, name)
     |      Forwards unknown attribute requests to VTK object.
     |  
     |  __init__(self, vtkobject)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

FUNCTIONS
    WrapDataObject(ds)
        Returns a Numpy friendly wrapper of a vtkDataObject.
    
    buffer_shared(...)
        Check if two objects share the same buffer, meaning that they point to the same block of memory.  An TypeError exception will be raised if either of the objects does not provide a buffer.
    
    numpyTovtkDataArray(array, name='numpy_array', array_type=None)
        Given a numpy array or a VTKArray and a name, returns a vtkDataArray.
        The resulting vtkDataArray will store a reference to the numpy array:
        the numpy array is released only when the vtkDataArray is destroyed.
    
    reshape_append_ones(a1, a2)
        Returns a list with the two arguments, any of them may be
        processed.  If the arguments are numpy.ndarrays, append 1s to the
        shape of the array with the smallest number of dimensions until
        the arrays have the same number of dimensions. Does nothing if the
        arguments are not ndarrays or the arrays have the same number of
        dimensions.
    
    vtkDataArrayToVTKArray(array, dataset=None)
        Given a vtkDataArray and a dataset owning it, returns a VTKArray.

DATA
    NoneArray = <paraview.vtk.numpy_interface.dataset_adapter.VTKNoneArray...

FILE
    /home/drishti/paraview/paraview_build/lib/python3.7/site-packages/vtkmodules/numpy_interface/dataset_adapter.py

### util

#### colors

#### keys

#### misc

#### numpy_support
NAME
    paraview.vtk.util.numpy_support

DESCRIPTION
    This module adds support to easily import and export NumPy
    (http://numpy.scipy.org) arrays into/out of VTK arrays.  The code is
    loosely based on TVTK (https://svn.enthought.com/enthought/wiki/TVTK).
    
    This code depends on an addition to the VTK data arrays made by Berk
    Geveci to make it support Python's buffer protocol (on Feb. 15, 2008).
    
    The main functionality of this module is provided by the two functions:
        numpy_to_vtk,
        vtk_to_numpy.
    
    
    Caveats:
    --------
    
     - Bit arrays in general do not have a numpy equivalent and are not
       supported.  Char arrays are also not easy to handle and might not
       work as you expect.  Patches welcome.
    
     - You need to make sure you hold a reference to a Numpy array you want
       to import into VTK.  If not you'll get a segfault (in the best case).
       The same holds in reverse when you convert a VTK array to a numpy
       array -- don't delete the VTK array.
    
    
    Created by Prabhu Ramachandran in Feb. 2008.

FUNCTIONS
    create_vtk_array(vtk_arr_type)
        Internal function used to create a VTK data array from another
        VTK array given the VTK array type.
    
    get_numpy_array_type(vtk_array_type)
        Returns a numpy array typecode given a VTK array type.
    
    get_vtk_array_type(numpy_array_type)
        Returns a VTK typecode given a numpy array.
    
    get_vtk_to_numpy_typemap()
        Returns the VTK array type to numpy array type mapping.
    
    numpy_to_vtk(num_array, deep=0, array_type=None)
        Converts a real numpy Array to a VTK array object.
        
        This function only works for real arrays.
        Complex arrays are NOT handled.  It also works for multi-component
        arrays.  However, only 1, and 2 dimensional arrays are supported.
        This function is very efficient, so large arrays should not be a
        problem.
        
        If the second argument is set to 1, the array is deep-copied from
        from numpy. This is not as efficient as the default behavior
        (shallow copy) and uses more memory but detaches the two arrays
        such that the numpy array can be released.
        
        WARNING: You must maintain a reference to the passed numpy array, if
        the numpy data is gc'd and VTK will point to garbage which will in
        the best case give you a segfault.
        
        Parameters:
        
        num_array
          a 1D or 2D, real numpy array.
    
    numpy_to_vtkIdTypeArray(num_array, deep=0)
    
    vtk_to_numpy(vtk_array)
        Converts a VTK data array to a numpy array.
        
        Given a subclass of vtkDataArray, this function returns an
        appropriate numpy array containing the same data -- it actually
        points to the same data.
        
        WARNING: This does not work for bit arrays.
        
        Parameters
        
        vtk_array
          The VTK data array to be converted.

DATA
    VTK_ID_TYPE_SIZE = 8
    VTK_LONG_TYPE_SIZE = 8

FILE
    /home/drishti/paraview/paraview_build/lib/python3.7/site-packages/vtkmodules/util/numpy_support.py




#### vtkAlgorithm

NAME
    paraview.vtk.util.vtkAlgorithm

CLASSES
    builtins.object
        VTKAlgorithm
    vtkmodules.vtkFiltersPython.vtkPythonAlgorithm(vtkmodules.vtkCommonExecutionModel.vtkAlgorithm)
        VTKPythonAlgorithmBase
    
    class VTKAlgorithm(builtins.object)
     |  VTKAlgorithm(nInputPorts=1, inputType='vtkDataSet', nOutputPorts=1, outputType='vtkPolyData')
     |  
     |  This is a superclass which can be derived to implement
     |  Python classes that work with vtkPythonAlgorithm. It implements
     |  Initialize(), ProcessRequest(), FillInputPortInformation() and
     |  FillOutputPortInformation().
     |  
     |  Initialize() sets the input and output ports based on data
     |  members.
     |  
     |  ProcessRequest() calls RequestXXX() methods to implement
     |  various pipeline passes.
     |  
     |  FillInputPortInformation() and FillOutputPortInformation() set
     |  the input and output types based on data members.
     |  
     |  Methods defined here:
     |  
     |  FillInputPortInformation(self, vtkself, port, info)
     |      Sets the required input type to InputType.
     |  
     |  FillOutputPortInformation(self, vtkself, port, info)
     |      Sets the default output type to OutputType.
     |  
     |  GetInputData(self, inInfo, i, j)
     |      Convenience method that returns an input data object
     |      given a vector of information objects and two indices.
     |  
     |  GetOutputData(self, outInfo, i)
     |      Convenience method that returns an output data object
     |      given an information object and an index.
     |  
     |  Initialize(self, vtkself)
     |      Sets up number of input and output ports based on
     |      NumberOfInputPorts and NumberOfOutputPorts.
     |  
     |  ProcessRequest(self, vtkself, request, inInfo, outInfo)
     |      Splits a request to RequestXXX() methods.
     |  
     |  RequestData(self, vtkself, request, inInfo, outInfo)
     |      Overwritten by subclass to execute the algorithm.
     |  
     |  RequestDataObject(self, vtkself, request, inInfo, outInfo)
     |      Overwritten by subclass to manage data object creation.
     |      There is not need to overwrite this class if the output can
     |      be created based on the OutputType data member.
     |  
     |  RequestInformation(self, vtkself, request, inInfo, outInfo)
     |      Overwritten by subclass to provide meta-data to downstream
     |      pipeline.
     |  
     |  RequestUpdateExtent(self, vtkself, request, inInfo, outInfo)
     |      Overwritten by subclass to modify data request going
     |      to upstream pipeline.
     |  
     |  __init__(self, nInputPorts=1, inputType='vtkDataSet', nOutputPorts=1, outputType='vtkPolyData')
     |      Sets up default NumberOfInputPorts, NumberOfOutputPorts,
     |      InputType and OutputType that are used by various initialization
     |      methods.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class VTKPythonAlgorithmBase(vtkmodules.vtkFiltersPython.vtkPythonAlgorithm)
     |  VTKPythonAlgorithmBase(nInputPorts=1, inputType='vtkDataSet', nOutputPorts=1, outputType='vtkPolyData')
     |  
     |  This is a superclass which can be derived to implement
     |  Python classes that act as VTK algorithms in a VTK pipeline.
     |  It implements ProcessRequest(), FillInputPortInformation() and
     |  FillOutputPortInformation().
     |  
     |  ProcessRequest() calls RequestXXX() methods to implement
     |  various pipeline passes.
     |  
     |  FillInputPortInformation() and FillOutputPortInformation() set
     |  the input and output types based on data members.
     |  
     |  Common use is something like this:
     |  
     |  class HDF5Source(VTKPythonAlgorithmBase):
     |      def __init__(self):
     |          VTKPythonAlgorithmBase.__init__(self,
     |              nInputPorts=0,
     |              nOutputPorts=1, outputType='vtkImageData')
     |  
     |      def RequestInformation(self, request, inInfo, outInfo):
     |          f = h5py.File("foo.h5", 'r')
     |          dims = f['RTData'].shape[::-1]
     |          info = outInfo.GetInformationObject(0)
     |          info.Set(vtkmodules.vtkCommonExecutionModel.vtkStreamingDemandDrivenPipeline.WHOLE_EXTENT(),
     |              (0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1), 6)
     |          return 1
     |  
     |      def RequestData(self, request, inInfo, outInfo):
     |          f = h5py.File("foo.h5", 'r')
     |          data = f['RTData'][:]
     |          output = dsa.WrapDataObject(vtkmodules.vtkCommonDataModel.vtkImageData.GetData(outInfo))
     |          output.SetDimensions(data.shape)
     |          output.PointData.append(data.flatten(), 'RTData')
     |          output.PointData.SetActiveScalars('RTData')
     |          return 1
     |  
     |  alg = HDF5Source()
     |  
     |  cf = vtkmodules.vtkFiltersCore.vtkContourFilter()
     |  cf.SetInputConnection(alg.GetOutputPort())
     |  cf.Update()
     |  
     |  Method resolution order:
     |      VTKPythonAlgorithmBase
     |      vtkmodules.vtkFiltersPython.vtkPythonAlgorithm
     |      vtkmodules.vtkCommonExecutionModel.vtkAlgorithm
     |      vtkmodules.vtkCommonCore.vtkObject
     |      vtkmodules.vtkCommonCore.vtkObjectBase
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  FillInputPortInformation(self, port, info)
     |      Sets the required input type to InputType.
     |  
     |  FillOutputPortInformation(self, port, info)
     |      Sets the default output type to OutputType.
     |  
     |  GetInputData(self, inInfo, i, j)
     |      Convenience method that returns an input data object
     |      given a vector of information objects and two indices.
     |  
     |  GetOutputData(self, outInfo, i)
     |      Convenience method that returns an output data object
     |      given an information object and an index.
     |  
     |  ProcessRequest(self, request, inInfo, outInfo)
     |      Splits a request to RequestXXX() methods.
     |  
     |  RequestData(self, request, inInfo, outInfo)
     |      Overwritten by subclass to execute the algorithm.
     |  
     |  RequestDataObject(self, request, inInfo, outInfo)
     |      Overwritten by subclass to manage data object creation.
     |      There is not need to overwrite this class if the output can
     |      be created based on the OutputType data member.
     |  
     |  RequestInformation(self, request, inInfo, outInfo)
     |      Overwritten by subclass to provide meta-data to downstream
     |      pipeline.
     |  
     |  RequestUpdateExtent(self, request, inInfo, outInfo)
     |      Overwritten by subclass to modify data request going
     |      to upstream pipeline.
     |  
     |  __init__(self, nInputPorts=1, inputType='vtkDataSet', nOutputPorts=1, outputType='vtkPolyData')
     |      Sets up default NumberOfInputPorts, NumberOfOutputPorts,
     |      InputType and OutputType that are used by various methods.
     |      Make sure to call this method from any subclass' __init__
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  InternalAlgorithm = <class 'paraview.vtk.util.vtkAlgorithm.VTKPythonAl...
     |      Internal class. Do not use.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from vtkmodules.vtkFiltersPython.vtkPythonAlgorithm:
     |  
     |  GetNumberOfGenerationsFromBase(...)
     |      V.GetNumberOfGenerationsFromBase(string) -> int
     |      C++: vtkIdType GetNumberOfGenerationsFromBase(const char *type)
     |          override;
     |      
     |      Given the name of a base class of this class type, return the
     |      distance of inheritance between this class type and the named
     |      class (how many generations of inheritance are there between this
     |      class and the named class). If the named class is not in this
     |      class's inheritance tree, return a negative value. Valid
     |      responses will always be nonnegative. This method works in
     |      combination with vtkTypeMacro found in vtkSetGet.h.
     |  
     |  GetNumberOfGenerationsFromBaseType(...)
     |      V.GetNumberOfGenerationsFromBaseType(string) -> int
     |      C++: static vtkIdType GetNumberOfGenerationsFromBaseType(
     |          const char *type)
     |      
     |      Given a the name of a base class of this class type, return the
     |      distance of inheritance between this class type and the named
     |      class (how many generations of inheritance are there between this
     |      class and the named class). If the named class is not in this
     |      class's inheritance tree, return a negative value. Valid
     |      responses will always be nonnegative. This method works in
     |      combination with vtkTypeMacro found in vtkSetGet.h.
     |  
     |  IsA(...)
     |      V.IsA(string) -> int
     |      C++: vtkTypeBool IsA(const char *type) override;
     |      
     |      Return 1 if this class is the same type of (or a subclass of) the
     |      named class. Returns 0 otherwise. This method works in
     |      combination with vtkTypeMacro found in vtkSetGet.h.
     |  
     |  IsTypeOf(...)
     |      V.IsTypeOf(string) -> int
     |      C++: static vtkTypeBool IsTypeOf(const char *type)
     |      
     |      Return 1 if this class type is the same type of (or a subclass
     |      of) the named class. Returns 0 otherwise. This method works in
     |      combination with vtkTypeMacro found in vtkSetGet.h.
     |  
     |  NewInstance(...)
     |      V.NewInstance() -> vtkPythonAlgorithm
     |      C++: vtkPythonAlgorithm *NewInstance()
     |  
     |  SafeDownCast(...)
     |      V.SafeDownCast(vtkObjectBase) -> vtkPythonAlgorithm
     |      C++: static vtkPythonAlgorithm *SafeDownCast(vtkObjectBase *o)
     |  
     |  SetNumberOfInputPorts(...)
     |      V.SetNumberOfInputPorts(int)
     |      C++: void SetNumberOfInputPorts(int n) override;
     |      
     |      Set the number of input ports used by the algorithm. This is made
     |      public so that it can be called from Python.
     |  
     |  SetNumberOfOutputPorts(...)
     |      V.SetNumberOfOutputPorts(int)
     |      C++: void SetNumberOfOutputPorts(int n) override;
     |      
     |      Set the number of output ports provided by the algorithm. This is
     |      made public so that it can be called from Python.
     |  
     |  SetPythonObject(...)
     |      V.SetPythonObject(PyObject)
     |      C++: void SetPythonObject(PyObject *obj)
     |      
     |      Specify the Python object to use to operate on the data. A
     |      reference will be taken on the object. This will also invoke
     |      Initialize() on the Python object, which is commonly used to set
     |      the number of input and output ports as well as perform tasks
     |      commonly performed in the constructor of C++ algorithm subclass.
     |  
     |  __delattr__(self, name, /)
     |      Implement delattr(self, name).
     |  
     |  __getattribute__(self, name, /)
     |      Return getattr(self, name).
     |  
     |  __repr__(self, /)
     |      Return repr(self).
     |  
     |  __setattr__(self, name, value, /)
     |      Implement setattr(self, name, value).
     |  
     |  __str__(self, /)
     |      Return str(self).
     |  
     |  ----------------------------------------------------------------------
     |  Static methods inherited from vtkmodules.vtkFiltersPython.vtkPythonAlgorithm:
     |  
     |  __new__(*args, **kwargs) from builtins.type
     |      Create and return a new object.  See help(type) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from vtkmodules.vtkFiltersPython.vtkPythonAlgorithm:
     |  
     |  __dict__
     |      Dictionary of attributes set by user.
     |  
     |  __this__
     |      Pointer to the C++ object.
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from vtkmodules.vtkFiltersPython.vtkPythonAlgorithm:
     |  
     |  __vtkname__ = 'vtkPythonAlgorithm'
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from vtkmodules.vtkCommonExecutionModel.vtkAlgorithm:
     |  
     |  AbortExecuteOff(...)
     |      V.AbortExecuteOff()
     |      C++: virtual void AbortExecuteOff()
     |      
     |      Set/Get the AbortExecute flag for the process object. Process
     |      objects may handle premature termination of execution in
     |      different ways.
     |  
     |  AbortExecuteOn(...)
     |      V.AbortExecuteOn()
     |      C++: virtual void AbortExecuteOn()
     |      
     |      Set/Get the AbortExecute flag for the process object. Process
     |      objects may handle premature termination of execution in
     |      different ways.
     |  
     |  AddInputConnection(...)
     |      V.AddInputConnection(int, vtkAlgorithmOutput)
     |      C++: virtual void AddInputConnection(int port,
     |          vtkAlgorithmOutput *input)
     |      V.AddInputConnection(vtkAlgorithmOutput)
     |      C++: virtual void AddInputConnection(vtkAlgorithmOutput *input)
     |      
     |      Add a connection to the given input port index.  See
     |      SetInputConnection() for details on input connections.  This
     |      method is the complement to RemoveInputConnection() in that it
     |      adds only the connection specified without affecting other
     |      connections.  Typical usage is
     |      
     |      * filter2->AddInputConnection(0, filter1->GetOutputPort(0)).
     |  
     |  AddInputDataObject(...)
     |      V.AddInputDataObject(int, vtkDataObject)
     |      C++: virtual void AddInputDataObject(int port,
     |          vtkDataObject *data)
     |      V.AddInputDataObject(vtkDataObject)
     |      C++: virtual void AddInputDataObject(vtkDataObject *data)
     |      
     |      Add the data-object as an input to this given port. This will add
     |      a new input connection on the specified port without affecting
     |      any existing connections on the same input port.
     |  
     |  CAN_HANDLE_PIECE_REQUEST(...)
     |      V.CAN_HANDLE_PIECE_REQUEST() -> vtkInformationIntegerKey
     |      C++: static vtkInformationIntegerKey *CAN_HANDLE_PIECE_REQUEST()
     |      
     |      Key that tells the pipeline that a particular algorithm can or
     |      cannot handle piece request. If a filter cannot handle piece
     |      requests and is asked for a piece, the executive will flag an
     |      error. If a structured data source cannot handle piece requests
     |      but can produce sub-extents (CAN_PRODUCE_SUB_EXTENT), the
     |      executive will use an extent translator to split the extent into
     |      pieces. Otherwise, if a source cannot handle piece requests, the
     |      executive will ask for the whole data for piece 0 and not execute
     |      the source for other pieces.\ingroup InformationKeys
     |  
     |  CAN_PRODUCE_SUB_EXTENT(...)
     |      V.CAN_PRODUCE_SUB_EXTENT() -> vtkInformationIntegerKey
     |      C++: static vtkInformationIntegerKey *CAN_PRODUCE_SUB_EXTENT()
     |      
     |      This key tells the executive that a particular output port is
     |      capable of producing an arbitrary subextent of the whole extent.
     |      Many image sources and readers fall into this category but some
     |      such as the legacy structured data readers cannot support this
     |      feature.\ingroup InformationKeys
     |  
     |  ConvertTotalInputToPortConnection(...)
     |      V.ConvertTotalInputToPortConnection(int, int, int)
     |      C++: void ConvertTotalInputToPortConnection(int ind, int &port,
     |          int &conn)
     |      
     |      Convenience routine to convert from a linear ordering of input
     |      connections to a port/connection pair.
     |  
     |  GetAbortExecute(...)
     |      V.GetAbortExecute() -> int
     |      C++: virtual vtkTypeBool GetAbortExecute()
     |      
     |      Set/Get the AbortExecute flag for the process object. Process
     |      objects may handle premature termination of execution in
     |      different ways.
     |  
     |  GetErrorCode(...)
     |      V.GetErrorCode() -> int
     |      C++: virtual unsigned long GetErrorCode()
     |      
     |      The error code contains a possible error that occurred while
     |      reading or writing the file.
     |  
     |  GetExecutive(...)
     |      V.GetExecutive() -> vtkExecutive
     |      C++: vtkExecutive *GetExecutive()
     |      
     |      Get this algorithm's executive.  If it has none, a default
     |      executive will be created.
     |  
     |  GetInformation(...)
     |      V.GetInformation() -> vtkInformation
     |      C++: virtual vtkInformation *GetInformation()
     |      
     |      Set/Get the information object associated with this algorithm.
     |  
     |  GetInputAlgorithm(...)
     |      V.GetInputAlgorithm(int, int, int) -> vtkAlgorithm
     |      C++: vtkAlgorithm *GetInputAlgorithm(int port, int index,
     |          int &algPort)
     |      V.GetInputAlgorithm(int, int) -> vtkAlgorithm
     |      C++: vtkAlgorithm *GetInputAlgorithm(int port, int index)
     |      V.GetInputAlgorithm() -> vtkAlgorithm
     |      C++: vtkAlgorithm *GetInputAlgorithm()
     |      
     |      Returns the algorithm and the output port index of that algorithm
     |      connected to a port-index pair.
     |  
     |  GetInputArrayInformation(...)
     |      V.GetInputArrayInformation(int) -> vtkInformation
     |      C++: vtkInformation *GetInputArrayInformation(int idx)
     |      
     |      Get the info object for the specified input array to this
     |      algorithm
     |  
     |  GetInputConnection(...)
     |      V.GetInputConnection(int, int) -> vtkAlgorithmOutput
     |      C++: vtkAlgorithmOutput *GetInputConnection(int port, int index)
     |      
     |      Get the algorithm output port connected to an input port.
     |  
     |  GetInputDataObject(...)
     |      V.GetInputDataObject(int, int) -> vtkDataObject
     |      C++: vtkDataObject *GetInputDataObject(int port, int connection)
     |      
     |      Get the data object that will contain the algorithm input for the
     |      given port and given connection.
     |  
     |  GetInputExecutive(...)
     |      V.GetInputExecutive(int, int) -> vtkExecutive
     |      C++: vtkExecutive *GetInputExecutive(int port, int index)
     |      V.GetInputExecutive() -> vtkExecutive
     |      C++: vtkExecutive *GetInputExecutive()
     |      
     |      Returns the executive associated with a particular input
     |      connection.
     |  
     |  GetInputInformation(...)
     |      V.GetInputInformation(int, int) -> vtkInformation
     |      C++: vtkInformation *GetInputInformation(int port, int index)
     |      V.GetInputInformation() -> vtkInformation
     |      C++: vtkInformation *GetInputInformation()
     |      
     |      Return the information object that is associated with a
     |      particular input connection. This can be used to get meta-data
     |      coming from the REQUEST_INFORMATION pass and set requests for the
     |      REQUEST_UPDATE_EXTENT pass. NOTE: Do not use this in any of the
     |      pipeline passes. Use the information objects passed as arguments
     |      instead.
     |  
     |  GetInputPortInformation(...)
     |      V.GetInputPortInformation(int) -> vtkInformation
     |      C++: vtkInformation *GetInputPortInformation(int port)
     |      
     |      Get the information object associated with an input port.  There
     |      is one input port per kind of input to the algorithm.  Each input
     |      port tells executives what kind of data and downstream requests
     |      this algorithm can handle for that input.
     |  
     |  GetNumberOfInputConnections(...)
     |      V.GetNumberOfInputConnections(int) -> int
     |      C++: int GetNumberOfInputConnections(int port)
     |      
     |      Get the number of inputs currently connected to a port.
     |  
     |  GetNumberOfInputPorts(...)
     |      V.GetNumberOfInputPorts() -> int
     |      C++: int GetNumberOfInputPorts()
     |      
     |      Get the number of input ports used by the algorithm.
     |  
     |  GetNumberOfOutputPorts(...)
     |      V.GetNumberOfOutputPorts() -> int
     |      C++: int GetNumberOfOutputPorts()
     |      
     |      Get the number of output ports provided by the algorithm.
     |  
     |  GetOutputDataObject(...)
     |      V.GetOutputDataObject(int) -> vtkDataObject
     |      C++: vtkDataObject *GetOutputDataObject(int port)
     |      
     |      Get the data object that will contain the algorithm output for
     |      the given port.
     |  
     |  GetOutputInformation(...)
     |      V.GetOutputInformation(int) -> vtkInformation
     |      C++: vtkInformation *GetOutputInformation(int port)
     |      
     |      Return the information object that is associated with a
     |      particular output port. This can be used to set meta-data coming
     |      during the REQUEST_INFORMATION. NOTE: Do not use this in any of
     |      the pipeline passes. Use the information objects passed as
     |      arguments instead.
     |  
     |  GetOutputPort(...)
     |      V.GetOutputPort(int) -> vtkAlgorithmOutput
     |      C++: vtkAlgorithmOutput *GetOutputPort(int index)
     |      V.GetOutputPort() -> vtkAlgorithmOutput
     |      C++: vtkAlgorithmOutput *GetOutputPort()
     |      
     |      Get a proxy object corresponding to the given output port of this
     |      algorithm.  The proxy object can be passed to another algorithm's
     |      SetInputConnection(), AddInputConnection(), and
     |      RemoveInputConnection() methods to modify pipeline connectivity.
     |  
     |  GetOutputPortInformation(...)
     |      V.GetOutputPortInformation(int) -> vtkInformation
     |      C++: vtkInformation *GetOutputPortInformation(int port)
     |      
     |      Get the information object associated with an output port.  There
     |      is one output port per output from the algorithm.  Each output
     |      port tells executives what kind of upstream requests this
     |      algorithm can handle for that output.
     |  
     |  GetProgress(...)
     |      V.GetProgress() -> float
     |      C++: virtual double GetProgress()
     |      
     |      Get the execution progress of a process object.
     |  
     |  GetProgressObserver(...)
     |      V.GetProgressObserver() -> vtkProgressObserver
     |      C++: virtual vtkProgressObserver *GetProgressObserver()
     |      
     |      If an ProgressObserver is set, the algorithm will report progress
     |      through it rather than directly. This means that it will call
     |      UpdateProgress() on the ProgressObserver rather than itself
     |      report it and set progress. This is most useful in situations
     |      where multiple threads are executing an algorithm at the same
     |      time and want to handle progress locally.
     |  
     |  GetProgressScale(...)
     |      V.GetProgressScale() -> float
     |      C++: virtual double GetProgressScale()
     |      
     |      Specify the shift and scale values to use to apply to the
     |      progress amount when `UpdateProgress` is called. By default shift
     |      is set to 0, and scale is set to 1.0. This is useful when the
     |      vtkAlgorithm instance is used as an internal algorithm to solve
     |      only a part of a whole problem.
     |      
     |      If calling on a internal vtkAlgorithm, make sure you take into
     |      consideration that values set of the outer vtkAlgorithm as well
     |      since the outer vtkAlgorithm itself may be nested in another
     |      algorithm.
     |      
     |      ote SetProgressShiftScale does not modify the MTime of the
     |      algorithm.
     |  
     |  GetProgressShift(...)
     |      V.GetProgressShift() -> float
     |      C++: virtual double GetProgressShift()
     |      
     |      Specify the shift and scale values to use to apply to the
     |      progress amount when `UpdateProgress` is called. By default shift
     |      is set to 0, and scale is set to 1.0. This is useful when the
     |      vtkAlgorithm instance is used as an internal algorithm to solve
     |      only a part of a whole problem.
     |      
     |      If calling on a internal vtkAlgorithm, make sure you take into
     |      consideration that values set of the outer vtkAlgorithm as well
     |      since the outer vtkAlgorithm itself may be nested in another
     |      algorithm.
     |      
     |      ote SetProgressShiftScale does not modify the MTime of the
     |      algorithm.
     |  
     |  GetProgressText(...)
     |      V.GetProgressText() -> string
     |      C++: virtual char *GetProgressText()
     |      
     |      Set the current text message associated with the progress state.
     |      This may be used by a calling process/GUI. Note: Because
     |      SetProgressText() is called from inside RequestData() it does not
     |      modify the algorithm object. Algorithms are not allowed to modify
     |      themselves from inside RequestData().
     |  
     |  GetReleaseDataFlag(...)
     |      V.GetReleaseDataFlag() -> int
     |      C++: virtual int GetReleaseDataFlag()
     |      
     |      Turn release data flag on or off for all output ports.
     |  
     |  GetTotalNumberOfInputConnections(...)
     |      V.GetTotalNumberOfInputConnections() -> int
     |      C++: int GetTotalNumberOfInputConnections()
     |      
     |      Get the total number of inputs for this algorithm
     |  
     |  GetUpdateExtent(...)
     |      V.GetUpdateExtent() -> (int, int, int, int, int, int)
     |      C++: int *GetUpdateExtent()
     |      V.GetUpdateExtent(int) -> (int, int, int, int, int, int)
     |      C++: int *GetUpdateExtent(int port)
     |      V.GetUpdateExtent(int, int, int, int, int, int)
     |      C++: void GetUpdateExtent(int &x0, int &x1, int &y0, int &y1,
     |          int &z0, int &z1)
     |      V.GetUpdateExtent(int, int, int, int, int, int, int)
     |      C++: void GetUpdateExtent(int port, int &x0, int &x1, int &y0,
     |          int &y1, int &z0, int &z1)
     |      V.GetUpdateExtent([int, int, int, int, int, int])
     |      C++: void GetUpdateExtent(int extent[6])
     |      V.GetUpdateExtent(int, [int, int, int, int, int, int])
     |      C++: void GetUpdateExtent(int port, int extent[6])
     |      
     |      These functions return the update extent for output ports that
     |      use 3D extents. Where port is not specified, it is assumed to be
     |      0.
     |  
     |  GetUpdateGhostLevel(...)
     |      V.GetUpdateGhostLevel() -> int
     |      C++: int GetUpdateGhostLevel()
     |      V.GetUpdateGhostLevel(int) -> int
     |      C++: int GetUpdateGhostLevel(int port)
     |      
     |      These functions return the update extent for output ports that
     |      use piece extents. Where port is not specified, it is assumed to
     |      be 0.
     |  
     |  GetUpdateNumberOfPieces(...)
     |      V.GetUpdateNumberOfPieces() -> int
     |      C++: int GetUpdateNumberOfPieces()
     |      V.GetUpdateNumberOfPieces(int) -> int
     |      C++: int GetUpdateNumberOfPieces(int port)
     |      
     |      These functions return the update extent for output ports that
     |      use piece extents. Where port is not specified, it is assumed to
     |      be 0.
     |  
     |  GetUpdatePiece(...)
     |      V.GetUpdatePiece() -> int
     |      C++: int GetUpdatePiece()
     |      V.GetUpdatePiece(int) -> int
     |      C++: int GetUpdatePiece(int port)
     |      
     |      These functions return the update extent for output ports that
     |      use piece extents. Where port is not specified, it is assumed to
     |      be 0.
     |  
     |  HasExecutive(...)
     |      V.HasExecutive() -> int
     |      C++: int HasExecutive()
     |      
     |      Check whether this algorithm has an assigned executive.  This
     |      will NOT create a default executive.
     |  
     |  INPUT_ARRAYS_TO_PROCESS(...)
     |      V.INPUT_ARRAYS_TO_PROCESS() -> vtkInformationInformationVectorKey
     |      C++: static vtkInformationInformationVectorKey *INPUT_ARRAYS_TO_PROCESS(
     |          )
     |      
     |      \ingroup InformationKeys
     |  
     |  INPUT_CONNECTION(...)
     |      V.INPUT_CONNECTION() -> vtkInformationIntegerKey
     |      C++: static vtkInformationIntegerKey *INPUT_CONNECTION()
     |      
     |      \ingroup InformationKeys
     |  
     |  INPUT_IS_OPTIONAL(...)
     |      V.INPUT_IS_OPTIONAL() -> vtkInformationIntegerKey
     |      C++: static vtkInformationIntegerKey *INPUT_IS_OPTIONAL()
     |      
     |      Keys used to specify input port requirements.\ingroup
     |      InformationKeys
     |  
     |  INPUT_IS_REPEATABLE(...)
     |      V.INPUT_IS_REPEATABLE() -> vtkInformationIntegerKey
     |      C++: static vtkInformationIntegerKey *INPUT_IS_REPEATABLE()
     |      
     |      \ingroup InformationKeys
     |  
     |  INPUT_PORT(...)
     |      V.INPUT_PORT() -> vtkInformationIntegerKey
     |      C++: static vtkInformationIntegerKey *INPUT_PORT()
     |      
     |      \ingroup InformationKeys
     |  
     |  INPUT_REQUIRED_DATA_TYPE(...)
     |      V.INPUT_REQUIRED_DATA_TYPE() -> vtkInformationStringVectorKey
     |      C++: static vtkInformationStringVectorKey *INPUT_REQUIRED_DATA_TYPE(
     |          )
     |      
     |      \ingroup InformationKeys
     |  
     |  INPUT_REQUIRED_FIELDS(...)
     |      V.INPUT_REQUIRED_FIELDS() -> vtkInformationInformationVectorKey
     |      C++: static vtkInformationInformationVectorKey *INPUT_REQUIRED_FIELDS(
     |          )
     |      
     |      \ingroup InformationKeys
     |  
     |  ModifyRequest(...)
     |      V.ModifyRequest(vtkInformation, int) -> int
     |      C++: virtual int ModifyRequest(vtkInformation *request, int when)
     |      
     |      This method gives the algorithm a chance to modify the contents
     |      of a request before or after (specified in the when argument) it
     |      is forwarded. The default implementation is empty. Returns 1 on
     |      success, 0 on failure. When can be either
     |      vtkExecutive::BeforeForward or vtkExecutive::AfterForward.
     |  
     |  PropagateUpdateExtent(...)
     |      V.PropagateUpdateExtent()
     |      C++: virtual void PropagateUpdateExtent()
     |      
     |      Propagate meta-data upstream.
     |  
     |  ReleaseDataFlagOff(...)
     |      V.ReleaseDataFlagOff()
     |      C++: void ReleaseDataFlagOff()
     |      
     |      Turn release data flag on or off for all output ports.
     |  
     |  ReleaseDataFlagOn(...)
     |      V.ReleaseDataFlagOn()
     |      C++: void ReleaseDataFlagOn()
     |      
     |      Turn release data flag on or off for all output ports.
     |  
     |  RemoveAllInputConnections(...)
     |      V.RemoveAllInputConnections(int)
     |      C++: virtual void RemoveAllInputConnections(int port)
     |      
     |      Removes all input connections.
     |  
     |  RemoveAllInputs(...)
     |      V.RemoveAllInputs()
     |      C++: void RemoveAllInputs()
     |      
     |      Remove all the input data.
     |  
     |  RemoveInputConnection(...)
     |      V.RemoveInputConnection(int, vtkAlgorithmOutput)
     |      C++: virtual void RemoveInputConnection(int port,
     |          vtkAlgorithmOutput *input)
     |      V.RemoveInputConnection(int, int)
     |      C++: virtual void RemoveInputConnection(int port, int idx)
     |      
     |      Remove a connection from the given input port index.  See
     |      SetInputConnection() for details on input connection.  This
     |      method is the complement to AddInputConnection() in that it
     |      removes only the connection specified without affecting other
     |      connections.  Typical usage is
     |      
     |      * filter2->RemoveInputConnection(0, filter1->GetOutputPort(0)).
     |  
     |  SetAbortExecute(...)
     |      V.SetAbortExecute(int)
     |      C++: virtual void SetAbortExecute(vtkTypeBool _arg)
     |      
     |      Set/Get the AbortExecute flag for the process object. Process
     |      objects may handle premature termination of execution in
     |      different ways.
     |  
     |  SetDefaultExecutivePrototype(...)
     |      V.SetDefaultExecutivePrototype(vtkExecutive)
     |      C++: static void SetDefaultExecutivePrototype(vtkExecutive *proto)
     |      
     |      If the DefaultExecutivePrototype is set, a copy of it is created
     |      in CreateDefaultExecutive() using NewInstance().
     |  
     |  SetExecutive(...)
     |      V.SetExecutive(vtkExecutive)
     |      C++: virtual void SetExecutive(vtkExecutive *executive)
     |      
     |      Set this algorithm's executive.  This algorithm is removed from
     |      any executive to which it has previously been assigned and then
     |      assigned to the given executive.
     |  
     |  SetInformation(...)
     |      V.SetInformation(vtkInformation)
     |      C++: virtual void SetInformation(vtkInformation *)
     |      
     |      Set/Get the information object associated with this algorithm.
     |  
     |  SetInputArrayToProcess(...)
     |      V.SetInputArrayToProcess(int, int, int, int, string)
     |      C++: virtual void SetInputArrayToProcess(int idx, int port,
     |          int connection, int fieldAssociation, const char *name)
     |      V.SetInputArrayToProcess(int, int, int, int, int)
     |      C++: virtual void SetInputArrayToProcess(int idx, int port,
     |          int connection, int fieldAssociation, int fieldAttributeType)
     |      V.SetInputArrayToProcess(int, vtkInformation)
     |      C++: virtual void SetInputArrayToProcess(int idx,
     |          vtkInformation *info)
     |      V.SetInputArrayToProcess(int, int, int, string, string)
     |      C++: virtual void SetInputArrayToProcess(int idx, int port,
     |          int connection, const char *fieldAssociation,
     |          const char *attributeTypeorName)
     |      
     |      Set the input data arrays that this algorithm will process.
     |      Specifically the idx array that this algorithm will process
     |      (starting from 0) is the array on port, connection with the
     |      specified association and name or attribute type (such as
     |      SCALARS). The fieldAssociation refers to which field in the data
     |      object the array is stored. See vtkDataObject::FieldAssociations
     |      for detail.
     |  
     |  SetInputConnection(...)
     |      V.SetInputConnection(int, vtkAlgorithmOutput)
     |      C++: virtual void SetInputConnection(int port,
     |          vtkAlgorithmOutput *input)
     |      V.SetInputConnection(vtkAlgorithmOutput)
     |      C++: virtual void SetInputConnection(vtkAlgorithmOutput *input)
     |      
     |      Set the connection for the given input port index.  Each input
     |      port of a filter has a specific purpose.  A port may have zero or
     |      more connections and the required number is specified by each
     |      filter.  Setting the connection with this method removes all
     |      other connections from the port.  To add more than one connection
     |      use AddInputConnection().
     |      
     |      * The input for the connection is the output port of another
     |      * filter, which is obtained with GetOutputPort().  Typical usage
     |        is
     |      
     |      * filter2->SetInputConnection(0, filter1->GetOutputPort(0)).
     |  
     |  SetInputDataObject(...)
     |      V.SetInputDataObject(int, vtkDataObject)
     |      C++: virtual void SetInputDataObject(int port,
     |          vtkDataObject *data)
     |      V.SetInputDataObject(vtkDataObject)
     |      C++: virtual void SetInputDataObject(vtkDataObject *data)
     |      
     |      Sets the data-object as an input on the given port index. Setting
     |      the input with this method removes all other connections from the
     |      port. Internally, this method creates a vtkTrivialProducer
     |      instance and sets that as the input-connection for the given
     |      port. It is safe to call this method repeatedly with the same
     |      input data object. The MTime of the vtkAlgorithm will not change
     |      unless the data object changed.
     |  
     |  SetProgress(...)
     |      V.SetProgress(float)
     |      C++: void SetProgress(double)
     |      
     |      `SetProgress` is deprecated. Subclasses should use
     |      `UpdateProgress` to report progress updates.
     |  
     |  SetProgressObserver(...)
     |      V.SetProgressObserver(vtkProgressObserver)
     |      C++: void SetProgressObserver(vtkProgressObserver *)
     |      
     |      If an ProgressObserver is set, the algorithm will report progress
     |      through it rather than directly. This means that it will call
     |      UpdateProgress() on the ProgressObserver rather than itself
     |      report it and set progress. This is most useful in situations
     |      where multiple threads are executing an algorithm at the same
     |      time and want to handle progress locally.
     |  
     |  SetProgressShiftScale(...)
     |      V.SetProgressShiftScale(float, float)
     |      C++: void SetProgressShiftScale(double shift, double scale)
     |      
     |      Specify the shift and scale values to use to apply to the
     |      progress amount when `UpdateProgress` is called. By default shift
     |      is set to 0, and scale is set to 1.0. This is useful when the
     |      vtkAlgorithm instance is used as an internal algorithm to solve
     |      only a part of a whole problem.
     |      
     |      If calling on a internal vtkAlgorithm, make sure you take into
     |      consideration that values set of the outer vtkAlgorithm as well
     |      since the outer vtkAlgorithm itself may be nested in another
     |      algorithm.
     |      
     |      ote SetProgressShiftScale does not modify the MTime of the
     |      algorithm.
     |  
     |  SetProgressText(...)
     |      V.SetProgressText(string)
     |      C++: void SetProgressText(const char *ptext)
     |      
     |      Set the current text message associated with the progress state.
     |      This may be used by a calling process/GUI. Note: Because
     |      SetProgressText() is called from inside RequestData() it does not
     |      modify the algorithm object. Algorithms are not allowed to modify
     |      themselves from inside RequestData().
     |  
     |  SetReleaseDataFlag(...)
     |      V.SetReleaseDataFlag(int)
     |      C++: virtual void SetReleaseDataFlag(int)
     |      
     |      Turn release data flag on or off for all output ports.
     |  
     |  Update(...)
     |      V.Update(int)
     |      C++: virtual void Update(int port)
     |      V.Update()
     |      C++: virtual void Update()
     |      V.Update(int, vtkInformationVector) -> int
     |      C++: virtual vtkTypeBool Update(int port,
     |          vtkInformationVector *requests)
     |      V.Update(vtkInformation) -> int
     |      C++: virtual vtkTypeBool Update(vtkInformation *requests)
     |      
     |      Bring this algorithm's outputs up-to-date.
     |  
     |  UpdateDataObject(...)
     |      V.UpdateDataObject()
     |      C++: virtual void UpdateDataObject()
     |      
     |      Create output object(s).
     |  
     |  UpdateExtent(...)
     |      V.UpdateExtent((int, int, int, int, int, int)) -> int
     |      C++: virtual int UpdateExtent(const int extents[6])
     |      
     |      Convenience method to update an algorithm after passing requests
     |      to its first output port. Supports extent request.
     |  
     |  UpdateExtentIsEmpty(...)
     |      V.UpdateExtentIsEmpty(vtkInformation, vtkDataObject) -> int
     |      C++: int UpdateExtentIsEmpty(vtkInformation *pinfo,
     |          vtkDataObject *output)
     |      V.UpdateExtentIsEmpty(vtkInformation, int) -> int
     |      C++: int UpdateExtentIsEmpty(vtkInformation *pinfo,
     |          int extentType)
     |      
     |      This detects when the UpdateExtent will generate no data This
     |      condition is satisfied when the UpdateExtent has zero volume
     |      (0,-1,...) or the UpdateNumberOfPieces is 0. The source uses this
     |      call to determine whether to call Execute.
     |  
     |  UpdateInformation(...)
     |      V.UpdateInformation()
     |      C++: virtual void UpdateInformation()
     |      
     |      Bring the algorithm's information up-to-date.
     |  
     |  UpdatePiece(...)
     |      V.UpdatePiece(int, int, int, (int, int, int, int, int, int))
     |          -> int
     |      C++: virtual int UpdatePiece(int piece, int numPieces,
     |          int ghostLevels, const int extents[6]=nullptr)
     |      
     |      Convenience method to update an algorithm after passing requests
     |      to its first output port. See documentation for Update(int port,
     |      vtkInformationVector* requests) for details. Supports piece and
     |      extent (optional) requests.
     |  
     |  UpdateProgress(...)
     |      V.UpdateProgress(float)
     |      C++: void UpdateProgress(double amount)
     |      
     |      Update the progress of the process object. If a ProgressMethod
     |      exists, executes it.  Then set the Progress ivar to amount. The
     |      parameter amount should range between (0,1).
     |  
     |  UpdateTimeStep(...)
     |      V.UpdateTimeStep(float, int, int, int, (int, int, int, int, int,
     |          int)) -> int
     |      C++: virtual int UpdateTimeStep(double time, int piece=-1,
     |          int numPieces=1, int ghostLevels=0,
     |          const int extents[6]=nullptr)
     |      
     |      Convenience method to update an algorithm after passing requests
     |      to its first output port. See documentation for Update(int port,
     |      vtkInformationVector* requests) for details. Supports time, piece
     |      (optional) and extent (optional) requests.
     |  
     |  UpdateWholeExtent(...)
     |      V.UpdateWholeExtent()
     |      C++: virtual void UpdateWholeExtent()
     |      
     |      Bring this algorithm's outputs up-to-date.
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from vtkmodules.vtkCommonExecutionModel.vtkAlgorithm:
     |  
     |  DEFAULT_PRECISION = 2
     |  
     |  DOUBLE_PRECISION = 1
     |  
     |  DesiredOutputPrecision = <class 'vtkmodules.vtkCommonExecutionModel.vt...
     |      int([x]) -> integer
     |      int(x, base=10) -> integer
     |      
     |      Convert a number or string to an integer, or return 0 if no arguments
     |      are given.  If x is a number, return x.__int__().  For floating point
     |      numbers, this truncates towards zero.
     |      
     |      If x is not a number or if base is given, then x must be a string,
     |      bytes, or bytearray instance representing an integer literal in the
     |      given base.  The literal can be preceded by '+' or '-' and be surrounded
     |      by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
     |      Base 0 means to interpret the base from the string as an integer literal.
     |      >>> int('0b100', base=0)
     |      4
     |  
     |  SINGLE_PRECISION = 0
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from vtkmodules.vtkCommonCore.vtkObject:
     |  
     |  AddObserver(...)
     |      V.AddObserver(int, function) -> int
     |      C++: unsigned long AddObserver(const char *event,
     |          vtkCommand *command, float priority=0.0f)
     |      
     |      Add an event callback function(vtkObject, int) for an event type.
     |      Returns a handle that can be used with RemoveEvent(int).
     |  
     |  BreakOnError(...)
     |      V.BreakOnError()
     |      C++: static void BreakOnError()
     |      
     |      This method is called when vtkErrorMacro executes. It allows the
     |      debugger to break on error.
     |  
     |  DebugOff(...)
     |      V.DebugOff()
     |      C++: virtual void DebugOff()
     |      
     |      Turn debugging output off.
     |  
     |  DebugOn(...)
     |      V.DebugOn()
     |      C++: virtual void DebugOn()
     |      
     |      Turn debugging output on.
     |  
     |  GetCommand(...)
     |      V.GetCommand(int) -> vtkCommand
     |      C++: vtkCommand *GetCommand(unsigned long tag)
     |      
     |      Allow people to add/remove/invoke observers (callbacks) to any
     |      VTK object.  This is an implementation of the subject/observer
     |      design pattern. An observer is added by specifying an event to
     |      respond to and a vtkCommand to execute. It returns an unsigned
     |      long tag which can be used later to remove the event or retrieve
     |      the command. When events are invoked, the observers are called in
     |      the order they were added. If a priority value is specified, then
     |      the higher priority commands are called first. A command may set
     |      an abort flag to stop processing of the event. (See vtkCommand.h
     |      for more information.)
     |  
     |  GetDebug(...)
     |      V.GetDebug() -> bool
     |      C++: bool GetDebug()
     |      
     |      Get the value of the debug flag.
     |  
     |  GetGlobalWarningDisplay(...)
     |      V.GetGlobalWarningDisplay() -> int
     |      C++: static int GetGlobalWarningDisplay()
     |      
     |      This is a global flag that controls whether any debug, warning or
     |      error messages are displayed.
     |  
     |  GetMTime(...)
     |      V.GetMTime() -> int
     |      C++: virtual vtkMTimeType GetMTime()
     |      
     |      Return this object's modified time.
     |  
     |  GlobalWarningDisplayOff(...)
     |      V.GlobalWarningDisplayOff()
     |      C++: static void GlobalWarningDisplayOff()
     |      
     |      This is a global flag that controls whether any debug, warning or
     |      error messages are displayed.
     |  
     |  GlobalWarningDisplayOn(...)
     |      V.GlobalWarningDisplayOn()
     |      C++: static void GlobalWarningDisplayOn()
     |      
     |      This is a global flag that controls whether any debug, warning or
     |      error messages are displayed.
     |  
     |  HasObserver(...)
     |      V.HasObserver(int, vtkCommand) -> int
     |      C++: vtkTypeBool HasObserver(unsigned long event, vtkCommand *)
     |      V.HasObserver(string, vtkCommand) -> int
     |      C++: vtkTypeBool HasObserver(const char *event, vtkCommand *)
     |      V.HasObserver(int) -> int
     |      C++: vtkTypeBool HasObserver(unsigned long event)
     |      V.HasObserver(string) -> int
     |      C++: vtkTypeBool HasObserver(const char *event)
     |      
     |      Allow people to add/remove/invoke observers (callbacks) to any
     |      VTK object.  This is an implementation of the subject/observer
     |      design pattern. An observer is added by specifying an event to
     |      respond to and a vtkCommand to execute. It returns an unsigned
     |      long tag which can be used later to remove the event or retrieve
     |      the command. When events are invoked, the observers are called in
     |      the order they were added. If a priority value is specified, then
     |      the higher priority commands are called first. A command may set
     |      an abort flag to stop processing of the event. (See vtkCommand.h
     |      for more information.)
     |  
     |  InvokeEvent(...)
     |      V.InvokeEvent(int, void) -> int
     |      C++: int InvokeEvent(unsigned long event, void *callData)
     |      V.InvokeEvent(string, void) -> int
     |      C++: int InvokeEvent(const char *event, void *callData)
     |      V.InvokeEvent(int) -> int
     |      C++: int InvokeEvent(unsigned long event)
     |      V.InvokeEvent(string) -> int
     |      C++: int InvokeEvent(const char *event)
     |      
     |      This method invokes an event and return whether the event was
     |      aborted or not. If the event was aborted, the return value is 1,
     |      otherwise it is 0.
     |  
     |  Modified(...)
     |      V.Modified()
     |      C++: virtual void Modified()
     |      
     |      Update the modification time for this object. Many filters rely
     |      on the modification time to determine if they need to recompute
     |      their data. The modification time is a unique monotonically
     |      increasing unsigned long integer.
     |  
     |  RemoveAllObservers(...)
     |      V.RemoveAllObservers()
     |      C++: void RemoveAllObservers()
     |  
     |  RemoveObserver(...)
     |      V.RemoveObserver(vtkCommand)
     |      C++: void RemoveObserver(vtkCommand *)
     |      V.RemoveObserver(int)
     |      C++: void RemoveObserver(unsigned long tag)
     |      
     |      Allow people to add/remove/invoke observers (callbacks) to any
     |      VTK object.  This is an implementation of the subject/observer
     |      design pattern. An observer is added by specifying an event to
     |      respond to and a vtkCommand to execute. It returns an unsigned
     |      long tag which can be used later to remove the event or retrieve
     |      the command. When events are invoked, the observers are called in
     |      the order they were added. If a priority value is specified, then
     |      the higher priority commands are called first. A command may set
     |      an abort flag to stop processing of the event. (See vtkCommand.h
     |      for more information.)
     |  
     |  RemoveObservers(...)
     |      V.RemoveObservers(int, vtkCommand)
     |      C++: void RemoveObservers(unsigned long event, vtkCommand *)
     |      V.RemoveObservers(string, vtkCommand)
     |      C++: void RemoveObservers(const char *event, vtkCommand *)
     |      V.RemoveObservers(int)
     |      C++: void RemoveObservers(unsigned long event)
     |      V.RemoveObservers(string)
     |      C++: void RemoveObservers(const char *event)
     |      
     |      Allow people to add/remove/invoke observers (callbacks) to any
     |      VTK object.  This is an implementation of the subject/observer
     |      design pattern. An observer is added by specifying an event to
     |      respond to and a vtkCommand to execute. It returns an unsigned
     |      long tag which can be used later to remove the event or retrieve
     |      the command. When events are invoked, the observers are called in
     |      the order they were added. If a priority value is specified, then
     |      the higher priority commands are called first. A command may set
     |      an abort flag to stop processing of the event. (See vtkCommand.h
     |      for more information.)
     |  
     |  SetDebug(...)
     |      V.SetDebug(bool)
     |      C++: void SetDebug(bool debugFlag)
     |      
     |      Set the value of the debug flag. A true value turns debugging on.
     |  
     |  SetGlobalWarningDisplay(...)
     |      V.SetGlobalWarningDisplay(int)
     |      C++: static void SetGlobalWarningDisplay(int val)
     |      
     |      This is a global flag that controls whether any debug, warning or
     |      error messages are displayed.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from vtkmodules.vtkCommonCore.vtkObjectBase:
     |  
     |  FastDelete(...)
     |      V.FastDelete()
     |      C++: virtual void FastDelete()
     |      
     |      Delete a reference to this object.  This version will not invoke
     |      garbage collection and can potentially leak the object if it is
     |      part of a reference loop.  Use this method only when it is known
     |      that the object has another reference and would not be collected
     |      if a full garbage collection check were done.
     |  
     |  GetAddressAsString(...)
     |      V.GetAddressAsString(string) -> string
     |      C++: const char *GetAddressAsString()
     |      
     |      Get address of C++ object in format 'Addr=%p' after casting to
     |      the specified type.  You can get the same information from o.__this__.
     |  
     |  GetClassName(...)
     |      V.GetClassName() -> string
     |      C++: const char *GetClassName()
     |      
     |      Return the class name as a string.
     |  
     |  GetIsInMemkind(...)
     |      V.GetIsInMemkind() -> bool
     |      C++: bool GetIsInMemkind()
     |      
     |      A local state flag that remembers whether this object lives in
     |      the normal or extended memory space.
     |  
     |  GetReferenceCount(...)
     |      V.GetReferenceCount() -> int
     |      C++: int GetReferenceCount()
     |      
     |      Return the current reference count of this object.
     |  
     |  GetUsingMemkind(...)
     |      V.GetUsingMemkind() -> bool
     |      C++: static bool GetUsingMemkind()
     |      
     |      A global state flag that controls whether vtkObjects are
     |      constructed in the usual way (the default) or within the extended
     |      memory space.
     |  
     |  InitializeObjectBase(...)
     |      V.InitializeObjectBase()
     |      C++: void InitializeObjectBase()
     |  
     |  Register(...)
     |      V.Register(vtkObjectBase)
     |      C++: virtual void Register(vtkObjectBase *o)
     |      
     |      Increase the reference count by 1.
     |  
     |  SetMemkindDirectory(...)
     |      V.SetMemkindDirectory(string)
     |      C++: static void SetMemkindDirectory(const char *directoryname)
     |      
     |      The name of a directory, ideally mounted -o dax, to memory map an
     |      extended memory space within. This must be called before any
     |      objects are constructed in the extended space. It can not be
     |      changed once setup.
     |  
     |  SetReferenceCount(...)
     |      V.SetReferenceCount(int)
     |      C++: void SetReferenceCount(int)
     |      
     |      Sets the reference count. (This is very dangerous, use with
     |      care.)
     |  
     |  UnRegister(...)
     |      V.UnRegister(vtkObjectBase)
     |      C++: virtual void UnRegister(vtkObjectBase *o)
     |      
     |      Decrease the reference count (release by another object). This
     |      has the same effect as invoking Delete() (i.e., it reduces the
     |      reference count by 1).

FILE
    /home/drishti/paraview/paraview_build/lib/python3.7/site-packages/vtkmodules/util/vtkAlgorithm.py


