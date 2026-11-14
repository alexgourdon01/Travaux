import gmsh
import meshio
import numpy as np
import sys

def gen_mesh(lc=0.01):
    gmsh.initialize()
    geom=gmsh.model.occ
    mesh=gmsh.model.mesh

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

    subdomains={}
    disk_center=np.asarray((-1., 0.0, 0.0))
    disk_radius=1.
    disk=geom.addDisk(disk_center[0], disk_center[1], disk_center[2], disk_radius, disk_radius)
    subdomains["D"] = disk # Disk

    rect_origin=(-1.0,-0.5,0.0)
    rect_length=(2.0,1.4,0)
    rect=geom.addRectangle(*(rect_origin+rect_length[:2]))
    subdomains["R"] = rect # rectangle

    subd,mapd=geom.fragment([(2,rect)], [(2,disk)])
    for d in subd:
        if (d in mapd[0]) and (d in mapd[1]):
            subdomains["I"] = d[1] # intersection
        else:
            if (d in mapd[0]):
                subdomains["R"] = d[1] # rectangle\disk
            if (d in mapd[1]):
                subdomains["D"] = d[1] # disk\rectangle

    geom.synchronize()
    gmsh.model.mesh.generate(2)

    gmsh.model.addPhysicalGroup(2, list(subdomains.values()),
                                tag=100, name="Domain")
    gmsh.model.addPhysicalGroup(2, [subdomains["D"], subdomains["I"]],
                                tag=110, name="Disk")
    gmsh.model.addPhysicalGroup(2, [subdomains["R"], subdomains["I"]],
                                tag=120, name="Rectangle")

    b_all, b_inside, b_outside = set(),set(),set()
    b_inside_disk, b_outside_disk = set(),set()
    b_inside_rect, b_outside_rect = set(),set()
    for b in gmsh.model.getBoundary([(2,subdomains["I"])]):
        be = abs(b[1])
        b_inside.add(be)
        if all(map(lambda x: np.sum((x-disk_center)**2)<(disk_radius-0.1*lc)**2,
                   gmsh.model.mesh.getNodes(1,be, includeBoundary=False)[1].reshape(-1,3))):
            b_inside_disk.add(be)
        else:
            b_inside_rect.add(be)
    for d in gmsh.model.getEntitiesForPhysicalGroup(2,100):
        for b in gmsh.model.getBoundary([(2,d)]):
            be = abs(b[1])
            b_all.add(be)
            if not be in b_inside:
                b_outside.add(be)
                if all(map(lambda x: np.sum((x-disk_center)**2)<(disk_radius+0.1*lc)**2,
                           gmsh.model.mesh.getNodes(1,be, includeBoundary=False)[1].reshape(-1,3))):
                    b_outside_disk.add(be)
                else:
                    b_outside_rect.add(be)

    gmsh.model.addPhysicalGroup(1, list(b_all),
                                tag=200, name="Boundaries")
    gmsh.model.addPhysicalGroup(1, list(b_inside),
                                tag=210, name="BoundariesInside")
    gmsh.model.addPhysicalGroup(1, list(b_inside_disk),
                                tag=211, name="BoundariesInsideD")
    gmsh.model.addPhysicalGroup(1, list(b_inside_rect),
                                tag=212, name="BoundariesInsideR")
    gmsh.model.addPhysicalGroup(1, list(b_outside),
                                tag=220, name="BoundariesOutside")
    gmsh.model.addPhysicalGroup(1, list(b_outside_disk),
                                tag=221, name="BoundariesOutsideD")
    gmsh.model.addPhysicalGroup(1, list(b_outside_rect),
                                tag=222, name="BoundariesOutsideR")

    # Whole mesh
    gmsh.write("dr.msh")

    # Disk only
    gmsh.model.removePhysicalGroups(list((2,_) for _ in (100,110,120)))
    gmsh.model.removePhysicalGroups(list((1,_) for _ in (200,210,211,212,220,221,222)))
    gmsh.model.addPhysicalGroup(2, [subdomains["D"], subdomains["I"]],
                                tag=100, name="Domain")
    gmsh.model.addPhysicalGroup(1, list(b_inside_disk),
                                tag=211, name="BoundariesInsideD")
    gmsh.model.addPhysicalGroup(1, list(b_inside_rect),
                                tag=212, name="BoundariesInsideR")
    gmsh.model.addPhysicalGroup(1, list(b_outside_disk),
                                tag=220, name="BoundariesOutside")
    gmsh.write("d.msh")

    # Rectangle only
    gmsh.model.removePhysicalGroups(list((2,_) for _ in (100,110,120)))
    gmsh.model.removePhysicalGroups(list((1,_) for _ in (200,210,211,212,220,221,222)))
    gmsh.model.addPhysicalGroup(2, [subdomains["R"], subdomains["I"]],
                                tag=100, name="Domain")
    gmsh.model.addPhysicalGroup(1, list(b_inside_disk),
                                tag=211, name="BoundariesInsideD")
    gmsh.model.addPhysicalGroup(1, list(b_inside_rect),
                                tag=212, name="BoundariesInsideR")
    gmsh.model.addPhysicalGroup(1, list(b_outside_rect),
                                tag=220, name="BoundariesOutside")
    gmsh.write("r.msh")

    gmsh.finalize()

if __name__=="__main__":
    gen_mesh()
