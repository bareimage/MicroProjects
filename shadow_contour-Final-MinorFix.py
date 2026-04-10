# -*- coding: utf-8 -*-

"""
Rhino 8 Python Script: Shadow Contour
Author: Bareimage (dot2dot) — MIT License
Shadow Boundary Solver — Final + Minor Fix

Explode for per-surface smoothing, interval-based edge extraction
from original polysurfaces, global join.
"""

import rhinoscriptsyntax as rs
import Rhino
import scriptcontext as sc
import Rhino.Geometry as rg
import math
import time


def ShadowContour():
    caster_ids = rs.GetObjects(
        "Select shadow-casting objects",
        rs.filter.surface | rs.filter.polysurface | rs.filter.mesh,
        preselect=True)
    if not caster_ids:
        return

    receiver_ids = rs.GetObjects(
        "Select shadow-receiving surfaces (can include casters + ground)",
        rs.filter.surface | rs.filter.polysurface | rs.filter.mesh)
    if not receiver_ids:
        return

    sun_vector = GetSunVector()
    if not sun_vector:
        return

    units = sc.doc.GetUnitSystemName(True, False, False, False)
    resolution = rs.GetReal(
        "Shadow resolution in {} (smaller = finer)".format(units),
        0.25, 0.001, 1000)
    if resolution is None:
        return

    fill_shadows = rs.GetString("Fill closed contours?", "Yes", ["Yes", "No"]) == "Yes"
    debug = rs.GetString("Debug mode?", "No", ["Yes", "No"]) == "Yes"

    rs.EnableRedraw(False)
    t0 = time.time()

    try:
        print("\n" + "=" * 60)
        print("  SHADOW CONTOUR — Resolution: {} {}".format(resolution, units))
        print("=" * 60)

        tol = sc.doc.ModelAbsoluteTolerance
        sun_n = rg.Vector3d(sun_vector)
        sun_n.Unitize()
        toward_sun = rg.Vector3d(-sun_n)
        toward_sun.Unitize()

        # Casters
        caster_list = []
        for cid in caster_ids:
            m = GetMesh(cid, 0)
            if m:
                caster_list.append(m)
        if not caster_list:
            print("No valid casters.")
            return

        caster = rg.Mesh()
        for m in caster_list:
            caster.Append(m)
        caster.Compact()
        print("Caster: {} faces".format(caster.Faces.Count))

        all_curve_ids = []
        all_fill_ids = []

        # Explode polysurfaces for per-surface smoothing (no hard edge bleed)
        exploded_ids = []
        temp_ids = []
        for rid in receiver_ids:
            if rs.IsPolysurface(rid):
                faces = rs.ExplodePolysurfaces(rid, False)
                if faces:
                    exploded_ids.extend(faces)
                    temp_ids.extend(faces)
                else:
                    exploded_ids.append(rid)
            else:
                exploded_ids.append(rid)

        print("  {} receivers -> {} surfaces".format(
            len(receiver_ids), len(exploded_ids)))

        recv_curves = []

        for ridx, rid in enumerate(exploded_ids):

            rm = GetMesh(rid, resolution)
            if not rm:
                continue

            nv = rm.Vertices.Count

            # Classify: normal test for self-shadow, ray cast for cast shadow
            t1 = time.time()
            rm.Normals.ComputeNormals()
            v_shad = [False] * nv
            v_dot = [0.0] * nv  # normal · toward_sun per vertex
            v_self = [False] * nv  # True if shadowed by normal (terminator)
            for vi in range(nv):
                nrm = rm.Normals[vi]
                dot = nrm.X * toward_sun.X + nrm.Y * toward_sun.Y + nrm.Z * toward_sun.Z
                v_dot[vi] = dot
                if dot <= 0:
                    v_shad[vi] = True
                    v_self[vi] = True
                    continue
                pt = rg.Point3d(rm.Vertices[vi])
                ray = rg.Ray3d(pt + toward_sun * (tol * 3), toward_sun)
                if rg.Intersect.Intersection.MeshRay(caster, ray) >= 0:
                    v_shad[vi] = True

            n_s = sum(1 for s in v_shad if s)
            if n_s == 0 or n_s == nv:
                continue

            # Crossings
            topo = rm.TopologyEdges
            topo_shad = {}
            topo_dot = {}
            topo_self = {}
            for tvi in range(rm.TopologyVertices.Count):
                mis = rm.TopologyVertices.MeshVertexIndices(tvi)
                if mis and len(mis) > 0:
                    mi = mis[0]
                    topo_shad[tvi] = v_shad[mi]
                    topo_dot[tvi] = v_dot[mi]
                    topo_self[tvi] = v_self[mi]

            edge_cross = {}
            for ei in range(topo.Count):
                evi = topo.GetTopologyVertices(ei)
                sa = topo_shad.get(evi.I, False)
                sb = topo_shad.get(evi.J, False)
                if sa == sb:
                    continue
                pa = rg.Point3d(rm.TopologyVertices[evi.I])
                pb = rg.Point3d(rm.TopologyVertices[evi.J])

                # Terminator edge: one vertex self-shadowed by normal,
                # other faces sun — interpolate where dot product = 0
                self_a = topo_self.get(evi.I, False)
                self_b = topo_self.get(evi.J, False)
                if self_a or self_b:
                    da = topo_dot.get(evi.I, 0.0)
                    db = topo_dot.get(evi.J, 0.0)
                    denom = da - db
                    if abs(denom) > 1e-12:
                        t_param = da / denom
                        t_param = max(0.0, min(1.0, t_param))
                    else:
                        t_param = 0.5
                    pt = rg.Point3d(
                        pa.X + t_param * (pb.X - pa.X),
                        pa.Y + t_param * (pb.Y - pa.Y),
                        pa.Z + t_param * (pb.Z - pa.Z))
                else:
                    # Cast shadow edge — use ray-based binary search
                    pt = BinSearch(pa, pb, sa, toward_sun, caster, tol)

                pt = PullToMesh(pt, rm, resolution, tol)
                edge_cross[ei] = pt

            if not edge_cross:
                continue

            # Chain
            face_edges = {}
            for ei in edge_cross:
                for fi in topo.GetConnectedFaces(ei):
                    if fi not in face_edges:
                        face_edges[fi] = []
                    face_edges[fi].append(ei)

            used = set()
            chains = []
            for start_ei in edge_cross:
                if start_ei in used:
                    continue
                chain = WalkChain(start_ei, edge_cross, topo, face_edges, used)
                if len(chain) > 1:
                    chains.append(rg.Polyline(chain))

            if not chains:
                continue

            # Extend open chain endpoints to surface boundary edges
            # so the contour starts/ends exactly on the brep edge.
            # Done pre-smoothing so the moving average includes the edge point.
            srf_brep = rs.coercebrep(rid)
            srf_edges = []
            if srf_brep:
                for sei in range(srf_brep.Edges.Count):
                    sec = srf_brep.Edges[sei].ToNurbsCurve()
                    if sec:
                        srf_edges.append(sec)

            if srf_edges:
                for ci in range(len(chains)):
                    pl = chains[ci]
                    if pl.IsClosed or pl.Count < 2:
                        continue
                    pts = [rg.Point3d(pl[k]) for k in range(pl.Count)]
                    for end_idx in [0, -1]:
                        pt = pts[end_idx]
                        best_dist = resolution * 2
                        best_pt = None
                        for sec in srf_edges:
                            ok, t = sec.ClosestPoint(pt)
                            if ok:
                                cp = sec.PointAt(t)
                                d = cp.DistanceTo(pt)
                                if d < best_dist:
                                    best_dist = d
                                    best_pt = cp
                        if best_pt:
                            pts[end_idx] = best_pt
                    chains[ci] = rg.Polyline(pts)

            # Join chains, subsample, create on-surface curve
            raw = []
            for pl in chains:
                if pl and pl.Count > 1:
                    crv = rg.PolylineCurve(pl)
                    if crv and crv.IsValid and crv.GetLength() > tol * 5:
                        raw.append(crv)

            surface_joined = rg.Curve.JoinCurves(raw, resolution * 1.2)
            if not surface_joined:
                surface_joined = raw

            for crv in surface_joined:
                if not crv or not crv.IsValid or crv.GetLength() < resolution * 3:
                    continue

                # Subsample points for on-surface interpolation
                length = crv.GetLength()
                n_sub = max(6, int(length / (resolution * 3)))
                n_sub = min(n_sub, 200)
                params = crv.DivideByCount(n_sub, True)
                if not params or len(params) < 3:
                    recv_curves.append(crv)
                    continue

                pts = [crv.PointAt(t) for t in params]

                # Create curve directly on the exploded surface
                try:
                    srf_crv_id = rs.AddInterpCrvOnSrf(rid, pts)
                    if srf_crv_id:
                        srf_crv = rs.coercecurve(srf_crv_id)
                        if srf_crv:
                            recv_curves.append(srf_crv.Duplicate())
                        rs.DeleteObject(srf_crv_id)
                    else:
                        recv_curves.append(crv)
                except:
                    recv_curves.append(crv)

        # Clean up exploded surfaces
        if temp_ids:
            rs.DeleteObjects(temp_ids)

        # Extract shadow boundary SEGMENTS from ALL receiver edges.
        # Polysurface internal edges: boundary if adjacent faces disagree.
        # Boundary/naked edges (1 face): boundary if that face is shadowed.
        for rid in receiver_ids:
            brep = rs.coercebrep(rid)
            if not brep:
                continue
            for ei in range(brep.Edges.Count):
                edge = brep.Edges[ei]
                adj_fi = edge.AdjacentFaces()
                if not adj_fi or len(adj_fi) == 0:
                    continue
                ec = edge.ToNurbsCurve()
                if not ec:
                    continue

                single_face = (len(adj_fi) == 1)

                # Sample edge
                dom = ec.Domain
                n_samples = max(8, int(ec.GetLength() / resolution))
                n_samples = min(n_samples, 64)
                samples = []
                for si in range(n_samples + 1):
                    t = dom.Min + (dom.Max - dom.Min) * si / n_samples
                    pt = ec.PointAt(t)

                    face_shad = []
                    for fi in adj_fi:
                        face = brep.Faces[fi]
                        ok, u, v = face.ClosestPoint(pt)
                        if not ok:
                            face_shad.append(False)
                            continue
                        nrm = face.NormalAt(u, v)
                        dot = (nrm.X * toward_sun.X +
                               nrm.Y * toward_sun.Y +
                               nrm.Z * toward_sun.Z)
                        if dot <= 0:
                            face_shad.append(True)
                        else:
                            ray = rg.Ray3d(pt + toward_sun * (tol * 3), toward_sun)
                            hit = rg.Intersect.Intersection.MeshRay(caster, ray) >= 0
                            face_shad.append(hit)

                    # Boundary-active:
                    # 2 faces: they disagree
                    # 1 face (naked edge): that face is shadowed
                    if single_face:
                        active = (len(face_shad) >= 1 and face_shad[0])
                    else:
                        active = (len(face_shad) >= 2 and face_shad[0] != face_shad[1])
                    samples.append((t, active))

                # Extract active intervals
                intervals = []
                in_active = False
                t_start = None
                for t, active in samples:
                    if active and not in_active:
                        t_start = t
                        in_active = True
                    elif not active and in_active:
                        intervals.append((t_start, t))
                        in_active = False
                if in_active:
                    intervals.append((t_start, samples[-1][0]))

                # Trim edge to active intervals
                edge_seg_count = 0
                for t0, t1 in intervals:
                    if t1 - t0 < 1e-9:
                        continue
                    seg = ec.Trim(rg.Interval(t0, t1))
                    if seg and seg.IsValid and seg.GetLength() > tol:
                        recv_curves.append(seg)
                        edge_seg_count += 1
                if debug and edge_seg_count > 0:
                    print("    edge {}: {} segments".format(ei, edge_seg_count))

        n_contours = len([c for c in recv_curves if c.GetLength() > resolution * 3])
        print("  {} total curves".format(len(recv_curves)))

        if not recv_curves:
            print("\nNo shadow contours found.")
            return

        # Join — contours + edges follow actual surface geometry
        joined = rg.Curve.JoinCurves(recv_curves, resolution * 2)
        if not joined:
            joined = recv_curves

        n_closed = sum(1 for c in joined if c.IsClosed)
        n_open = sum(1 for c in joined if not c.IsClosed)
        print("  joined: {} closed, {} open".format(n_closed, n_open))

        for crv in joined:
            if not crv or crv.GetLength() < resolution * 3:
                continue
            if crv.IsClosed and crv.GetLength() < resolution * 40:
                continue
            bb = crv.GetBoundingBox(True)
            if bb.IsValid and bb.Diagonal.Length < resolution * 8:
                continue

            # Close nearly-closed curves
            if not crv.IsClosed:
                gap = crv.PointAtStart.DistanceTo(crv.PointAtEnd)
                length = crv.GetLength()
                if gap < length * 0.05 or gap < resolution * 4:
                    crv.MakeClosed(gap + tol)

            oid = sc.doc.Objects.AddCurve(crv)
            if oid:
                all_curve_ids.append(oid)

        if not all_curve_ids:
            print("\nNo shadow contours found.")
            return

        layer = "Shadow_Contours"
        if not rs.IsLayer(layer):
            rs.AddLayer(layer, (40, 40, 40))
        rs.ObjectLayer(all_curve_ids, layer)

        if fill_shadows:
            for cid in all_curve_ids:
                if rs.IsCurveClosed(cid) and rs.IsCurvePlanar(cid):
                    try:
                        area = rs.CurveArea(cid)
                        if area and area[0] > tol * 200:
                            srf = rs.AddPlanarSrf(cid)
                            if srf:
                                all_fill_ids.extend(
                                    srf if isinstance(srf, list) else [srf])
                    except:
                        pass
            if all_fill_ids:
                fl = "Shadow_Fill"
                if not rs.IsLayer(fl):
                    rs.AddLayer(fl, (80, 80, 80))
                rs.ObjectLayer(all_fill_ids, fl)

        elapsed = time.time() - t0
        print("\nDone in {:.1f}s — {} contours, {} fills".format(
            elapsed, len(all_curve_ids), len(all_fill_ids)))

    except Exception as e:
        print("Error: {}".format(e))
        import traceback
        traceback.print_exc()
    finally:
        rs.EnableRedraw(True)


# ──────────────────────────────────────────────────────────────

def SmoothOnMesh(crv, mesh, resolution, tol):
    """
    Moving-average filter on the raw polyline.
    
    The crossing points zigzag left/right of the true shadow boundary
    because they're constrained to mesh edges. On a UV grid mesh this
    creates a regular staircase. A sliding-window average over several
    face widths finds the median path between the zigzag extremes.
    
    Subsampling (every Nth point) does NOT work — it picks points still
    on the staircase. Rebuild does NOT work — too few CPs destroys the
    curve, too many preserves the staircase. Only averaging works.
    """
    if not crv or not crv.IsValid:
        return crv
    length = crv.GetLength()
    if length < resolution * 2:
        return crv

    is_closed = crv.IsClosed

    # Dense sample: ~2 points per mesh face
    n_pts = max(10, int(length / (resolution * 0.5)))
    n_pts = min(n_pts, 4000)
    params = crv.DivideByCount(n_pts, True)
    if not params or len(params) < 6:
        return crv

    raw = [crv.PointAt(t) for t in params]
    n = len(raw)

    # Window: span ~6 mesh faces worth of samples
    # Each face is ~resolution wide, samples are spaced ~resolution/2
    # So 6 faces = ~12 samples
    win_samples = max(5, int(6.0 * resolution / (length / n)))
    if win_samples % 2 == 0:
        win_samples += 1
    half = win_samples // 2

    # Moving average
    smoothed = []
    for i in range(n):
        sx, sy, sz = 0.0, 0.0, 0.0
        count = 0
        for j in range(i - half, i + half + 1):
            if is_closed:
                idx = j % n
            else:
                idx = max(0, min(j, n - 1))
            sx += raw[idx].X
            sy += raw[idx].Y
            sz += raw[idx].Z
            count += 1
        smoothed.append(rg.Point3d(sx / count, sy / count, sz / count))

    # Subsample the smoothed points for the interpolation
    # (no need for dense points after averaging)
    step = max(1, int(n / max(6, int(length / (resolution * 3)))))
    sub = smoothed[::step]
    if smoothed[-1].DistanceTo(sub[-1]) > tol:
        sub.append(smoothed[-1])

    if len(sub) < 3:
        return crv

    if is_closed and sub[0].DistanceTo(sub[-1]) > tol:
        sub.append(sub[0])

    fitted = rg.Curve.CreateInterpolatedCurve(
        sub, 3, rg.CurveKnotStyle.ChordSquareRoot)
    if fitted and fitted.IsValid:
        if is_closed and not fitted.IsClosed:
            fitted.MakeClosed(max(resolution, tol * 10))
        return fitted
    return crv


def PullToMesh(pt, mesh, resolution, tol):
    mp = mesh.ClosestMeshPoint(pt, max(resolution * 1.5, tol * 10))
    return mp.Point if mp else pt


def WalkChain(start_ei, edge_cross, topo, face_edges, used):
    forward = [edge_cross[start_ei]]
    used.add(start_ei)
    cur = start_ei
    is_loop = False
    for _ in range(len(edge_cross) + 1):
        nxt = _find_next(cur, topo, face_edges, used)
        if nxt is None:
            # Check if we can close back to start
            if cur != start_ei and _shares_face(cur, start_ei, topo, face_edges):
                is_loop = True
            break
        forward.append(edge_cross[nxt])
        used.add(nxt)
        cur = nxt

    backward = []
    if not is_loop:
        cur = start_ei
        for _ in range(len(edge_cross) + 1):
            nxt = _find_next(cur, topo, face_edges, used)
            if nxt is None:
                break
            backward.append(edge_cross[nxt])
            used.add(nxt)
            cur = nxt

    if backward:
        backward.reverse()
        chain = backward + forward
    else:
        chain = forward

    # Close loop by appending first point
    if is_loop and len(chain) > 2:
        chain.append(chain[0])

    return chain


def _shares_face(ei_a, ei_b, topo, face_edges):
    """Check if two edges share a face that has both in face_edges."""
    for fi in topo.GetConnectedFaces(ei_a):
        if fi not in face_edges:
            continue
        if ei_b in face_edges[fi]:
            return True
    return False


def _find_next(cur_ei, topo, face_edges, used):
    for fi in topo.GetConnectedFaces(cur_ei):
        if fi not in face_edges:
            continue
        for nei in face_edges[fi]:
            if nei != cur_ei and nei not in used:
                return nei
    return None


def BinSearch(pa, pb, a_shad, toward_sun, caster, tol):
    off = tol * 4
    for _ in range(14):
        mid = (pa + pb) * 0.5
        ray = rg.Ray3d(mid + toward_sun * off, toward_sun)
        hit = rg.Intersect.Intersection.MeshRay(caster, ray) >= 0
        if hit == a_shad:
            pa = mid
        else:
            pb = mid
    return (pa + pb) * 0.5


def GetMesh(obj_id, target_edge):
    if rs.IsMesh(obj_id):
        mesh = rs.coercemesh(obj_id)
        if mesh and mesh.FaceNormals.Count == 0:
            mesh.FaceNormals.ComputeFaceNormals()
        return mesh
    brep = rs.coercebrep(obj_id)
    if not brep:
        return None
    bbox = brep.GetBoundingBox(True)
    size = bbox.Diagonal.Length
    params = rg.MeshingParameters()
    if target_edge > 0:
        params.MaximumEdgeLength = target_edge
        params.Tolerance = target_edge * 0.3
        params.GridAngle = math.radians(8)
        params.MinimumEdgeLength = target_edge * 0.1
        params.GridMinCount = max(4, int(size / target_edge / 2))
    else:
        params.Tolerance = sc.doc.ModelAbsoluteTolerance * 1.5
        params.MaximumEdgeLength = size * 0.05
        params.GridAngle = math.radians(12)
    params.RefineGrid = True
    params.SimplePlanes = False
    meshes = rg.Mesh.CreateFromBrep(brep, params)
    if not meshes:
        meshes = rg.Mesh.CreateFromBrep(brep, rg.MeshingParameters.Default)
    if not meshes:
        return None
    c = rg.Mesh()
    for m in meshes:
        if m:
            c.Append(m)
    c.Compact()
    c.Normals.ComputeNormals()
    c.FaceNormals.ComputeFaceNormals()
    c.UnifyNormals()
    c.Weld(math.radians(15))
    return c


def GetSunVector():
    choice = rs.GetString("Sun direction", "RhinoSun",
                          ["RhinoSun", "Manual", "Vertical", "Angle"])
    if choice == "RhinoSun":
        try:
            sun = sc.doc.Lights.Sun
            if sun.Enabled:
                vec = sun.Vector
                if vec.IsValid and vec.Length > 0:
                    print("  Using Rhino Sun: alt={:.1f} azi={:.1f}".format(
                        sun.Altitude, sun.Azimuth))
                    return vec
        except:
            pass
        print("  Using default sun.")
        vec = rg.Vector3d(1, 1, -1)
        vec.Unitize()
        return vec
    elif choice == "Manual":
        pt1 = rs.GetPoint("Sun position")
        if not pt1:
            return None
        pt2 = rs.GetPoint("Target", base_point=pt1)
        if not pt2:
            return None
        vec = pt2 - pt1
        vec.Unitize()
        return vec
    elif choice == "Vertical":
        return rg.Vector3d(0, 0, -1)
    elif choice == "Angle":
        alt = rs.GetReal("Altitude (0-90)", 45, 0, 90)
        azi = rs.GetReal("Azimuth (0-360, 0=N)", 135, 0, 360)
        if alt is None or azi is None:
            return None
        ar = math.radians(90 - alt)
        azr = math.radians(azi)
        vec = rg.Vector3d(math.sin(ar) * math.sin(azr),
                          math.sin(ar) * math.cos(azr),
                          -math.cos(ar))
        vec.Unitize()
        return vec
    return None


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SHADOW CONTOUR")
    print("=" * 60)
    ShadowContour()
