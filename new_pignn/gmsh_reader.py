from netgen.meshing import *


def _get_element_dim(eltype):
    if eltype == 15:
        return 0
    if eltype in {1, 8}:
        return 1
    if eltype in {2, 3, 9, 16}:
        return 2
    if eltype in {4, 5, 6, 7, 11, 17, 18, 19}:
        return 3
    return None


def _infer_name_dim(name, meshdim):
    lower_name = name.lower()
    if lower_name.startswith(("point", "vertex", "node")):
        return 0
    if lower_name.startswith(("edge", "line", "curve", "wire")):
        return 1
    if lower_name.startswith(("face", "surf", "surface", "boundary", "bnd", "shell")):
        return 2
    if lower_name.startswith(("solid", "volume", "domain", "body", "material", "mat")):
        return 3
    return meshdim


def _parse_physical_name(line, meshdim, physical_tag_dims):
    parts = line.strip().split(maxsplit=2)
    if len(parts) >= 3:
        dim = int(parts[0])
        tag = int(parts[1])
        name = parts[2]
    elif len(parts) == 2:
        tag = int(parts[0])
        name = parts[1]
        dims = physical_tag_dims.get(tag, set())
        if len(dims) == 1:
            dim = next(iter(dims))
        else:
            dim = _infer_name_dim(name, meshdim)
    else:
        raise ValueError("Invalid $PhysicalNames entry: {!r}".format(line.rstrip("\n")))

    if name.startswith('"') and name.endswith('"'):
        name = name[1:-1]

    return dim, tag, name

def ReadGmsh(filename):
    if not filename.endswith(".msh"):
        filename += ".msh"
    meshdim = 1
    physical_tag_dims = {}
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line == "":
                break
            parts = line.split()
            if parts and parts[0] == "$Elements":
                break

        nelem = int(f.readline().split()[0])
        for i in range(nelem):
            line = f.readline()
            parts = line.split()
            if not parts:
                continue

            eltype = int(parts[1])
            element_dim = _get_element_dim(eltype)
            if element_dim is not None:
                meshdim = max(meshdim, element_dim)

            numtags = int(parts[2])
            if numtags > 0:
                physical_tag_dims.setdefault(int(parts[3]), set()).add(element_dim)

    f = open(filename, 'r')
    mesh = Mesh(dim=meshdim)

    pointmap = {}
    facedescriptormap = {}
    namemap = { 0 : { 0 : "default" },
                1: { 0 : "default" },
                2: { 0 : "default" },
                3: { 0 : "default" } }
    materialmap = {}
    bbcmap = {}
    bbbcmap = {}
    

    segm = 1
    trig = 2
    quad = 3
    tet = 4
    hex = 5
    prism = 6
    pyramid = 7
    segm3 = 8      # 2nd order line
    trig6 = 9      # 2nd order trig
    tet10 = 11     # 2nd order tet
    point = 15
    quad8 = 16     # 2nd order quad
    hex20 = 17     # 2nd order hex
    prism15 = 18   # 2nd order prism
    pyramid13 = 19 # 2nd order pyramid
    segms = [segm, segm3]
    trigs = [trig, trig6]
    quads = [quad, quad8]
    tets = [tet, tet10]
    hexes = [hex, hex20]
    prisms = [prism, prism15]
    pyramids = [pyramid, pyramid13]
    elem0d = [point]
    elem1d = segms
    elem2d = trigs + quads
    elem3d = tets + hexes + prisms + pyramids

    num_nodes_map = { segm : 2,
                      trig : 3,
                      quad : 4,
                      tet : 4,
                      hex : 8,
                      prism : 6,
                      pyramid : 5,
                      segm3 : 3,
                      trig6 : 6,
                      tet10 : 10,
                      point : 1,
                      quad8 : 8,
                      hex20 : 20,
                      prism15 : 18,
                      pyramid13 : 19 }

    while True:
        line = f.readline()
        if line == "":
            break
        parts = line.split()
        if not parts:
            continue

        if parts[0] == "$PhysicalNames":
            print('WARNING: Physical groups detected - Be sure to define them for every geometrical entity.')
            numnames = int(f.readline())
            for i in range(numnames):
                line = f.readline()
                dim, tag, name = _parse_physical_name(line, meshdim, physical_tag_dims)
                namemap[dim][tag] = name

        if parts[0] == "$Nodes":
            num = int(f.readline().split()[0])
            for i in range(num):
                line = f.readline()
                nodenum, x, y, z = line.split()[0:4]
                pnum = mesh.Add(MeshPoint(Pnt(float(x), float(y), float(z))))
                pointmap[int(nodenum)] = pnum

        if parts[0] == "$Elements":
            num = int(f.readline().split()[0])

            for i in range(num):
                line = f.readline().split()
                elmnum = int(line[0])
                elmtype = int(line[1])
                numtags = int(line[2])
                # the first tag is the physical group nr, the second tag is the group nr of the dim
                tags = [int(line[3 + k]) for k in range(numtags)]

                if elmtype not in num_nodes_map:
                    raise Exception("element type", elmtype, "not implemented")
                num_nodes = num_nodes_map[elmtype]

                nodenums = line[3 + numtags:3 + numtags + num_nodes]
                nodenums2 = [pointmap[int(nn)] for nn in nodenums]


                if elmtype in elem0d:
                    if meshdim == 3:
                        if tags[1] in bbbcmap:
                            index = bbbcmap[tags[1]]
                        else:
                            index = len(bbbcmap) + 1
                            if len(namemap):
                                mesh.SetCD3Name(index, namemap[0][tags[0]])
                            else:
                                mesh.SetCD3Name(index, "point" + str(tags[1]))
                            bbbcmap[tags[1]] = index
                    elif meshdim == 2:
                        if tags[1] in bbcmap:
                            index = bbcmap[tags[1]]
                        else:
                            index = len(bbcmap) + 1
                            if len(namemap):
                                mesh.SetCD2Name(index, namemap[0][tags[0]])
                            else:
                                mesh.SetCD2Name(index, "point" + str(tags[1]))
                            bbcmap[tags[1]] = index
                    else:
                        if tags[1] in facedescriptormap.keys():
                            index = facedescriptormap[tags[1]]
                        else:
                            index = len(facedescriptormap) + 1
                            fd = FaceDescriptor(bc=index)
                            if len(namemap):
                                fd.bcname = namemap[0][tags[0]]
                            else:
                                fd.bcname = 'point' + str(tags[1])
                            mesh.SetBCName(index - 1, fd.bcname)
                            mesh.Add(fd)
                            facedescriptormap[tags[1]] = index

                    mesh.Add(Element0D(nodenums2[0], index=index))

                
                if elmtype in elem1d:
                    if meshdim == 3:
                        if tags[1] in bbcmap:
                            index = bbcmap[tags[1]]
                        else:
                            index = len(bbcmap) + 1
                            if len(namemap):
                                mesh.SetCD2Name(index, namemap[1][tags[0]])
                            else:
                                mesh.SetCD2Name(index, "line" + str(tags[1]))
                            bbcmap[tags[1]] = index

                    elif meshdim == 2:
                        if tags[1] in facedescriptormap.keys():
                            index = facedescriptormap[tags[1]]
                        else:
                            index = len(facedescriptormap) + 1
                            fd = FaceDescriptor(bc=index)
                            if len(namemap):
                                fd.bcname = namemap[1][tags[0]]
                            else:
                                fd.bcname = 'line' + str(tags[1])
                            mesh.SetBCName(index - 1, fd.bcname)
                            mesh.Add(fd)
                            facedescriptormap[tags[1]] = index
                    else:
                        if tags[1] in materialmap:
                            index = materialmap[tags[1]]
                        else:
                            index = len(materialmap) + 1
                            if len(namemap):
                                mesh.SetMaterial(index, namemap[1][tags[0]])
                            else:
                                mesh.SetMaterial(index, "line" + str(tags[1]))
                            materialmap[tags[1]] = index

                    mesh.Add(Element1D(index=index, vertices=nodenums2))

                if elmtype in elem2d:  # 2d elements
                    if meshdim == 3:
                        if tags[1] in facedescriptormap.keys():
                            index = facedescriptormap[tags[1]]
                        else:
                            index = len(facedescriptormap) + 1
                            fd = FaceDescriptor(bc=index)
                            if len(namemap):
                                fd.bcname = namemap[2][tags[0]]
                            else:
                                fd.bcname = "surf" + str(tags[1])
                            mesh.SetBCName(index - 1, fd.bcname)
                            mesh.Add(fd)
                            facedescriptormap[tags[1]] = index
                    else:
                        if tags[1] in materialmap:
                            index = materialmap[tags[1]]
                        else:
                            index = len(materialmap) + 1
                            if len(namemap):
                                mesh.SetMaterial(index, namemap[2][tags[0]])
                            else:
                                mesh.SetMaterial(index, "surf" + str(tags[1]))
                            materialmap[tags[1]] = index

                    if elmtype in trigs:
                        ordering = [i for i in range(3)]
                        if elmtype == trig6:
                            ordering += [4,5,3]
                    if elmtype in quads:
                        ordering = [i for i in range(4)]
                        if elmtype == quad8:
                            ordering += [4, 6, 7, 5]
                    mesh.Add(Element2D(index, [nodenums2[i] for i in ordering]))

                if elmtype in elem3d:  # volume elements
                    if tags[1] in materialmap:
                        index = materialmap[tags[1]]
                    else:
                        index = len(materialmap) + 1
                        if len(namemap):
                            mesh.SetMaterial(index, namemap[3][tags[0]])
                        else:
                            mesh.SetMaterial(index, "vol" + str(tags[1]))
                        materialmap[tags[1]] = index

                    nodenums2 = [pointmap[int(nn)] for nn in nodenums]

                    if elmtype in tets:
                        ordering = [0,1,2,3]
                        if elmtype == tet10:
                            ordering += [4,6,7,5,9,8]
                    elif elmtype in hexes:
                        ordering = [0,1,5,4,3,2,6,7]
                        if elmtype == hex20:
                            ordering += [8,16,10,12,13,19,15,14,9,11,18,17]
                    elif elmtype in prisms:
                        ordering = [0,2,1,3,5,4]
                        if elmtype == prism15:
                            ordering += [7,6,9,8,11,10,13,12,14]
                    elif elmtype in pyramids:
                        ordering = [3,2,1,0,4]
                        if elmtype == pyramid13:
                            ordering += [10,5,6,8,12,11,9,7]
                    mesh.Add(Element3D(index, [nodenums2[i] for i in ordering]))

    return mesh
