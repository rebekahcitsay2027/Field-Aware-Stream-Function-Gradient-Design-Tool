FeatureScript 2985;
import(path : "onshape/std/common.fs", version : "2985.0");
annotation { "Feature Type Name" : "Path from Points", "Feature Type Description" : "" }
export const pathFromPoints = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        annotation { "Name" : "Table" }
        definition.pointData is TableData;

        annotation { "Name" : "Sparse Points" }
        definition.enableSparse is boolean;

        annotation { "Group Name" : "Sparse Points", "Collapsed By Default" : true}
        {
            annotation { "Name" : "Minimum Distance" }
            isLength(definition.minDist, { (millimeter) : [ 0.001, 5, 10000 ] } as LengthBoundSpec );

            annotation { "Name" : "Maximum Distance" }
            isLength(definition.maxDist, LENGTH_BOUNDS);

            annotation { "Name" : "Minimum Angle Deviation" }
            isAngle(definition.minAngle, { (degree) : [ 0.01, 1, 15 ] } as AngleBoundSpec );
        }

        annotation { "Group Name" : "Preview", "Collapsed By Default" : true }
        {
            annotation { "Name" : "Preview Points" }
            definition.preview is boolean;

            annotation { "Name" : "Preview Raw" }
            definition.previewRaw is boolean;
        }

        annotation { "Name" : "Debug" }
        definition.debug is boolean;
    }
    {
        var pointData = definition.pointData.csvData;
        var nRows = size(pointData);

        var cx = 0;
        var cy = 1;
        var cz = 2;

        // ===== CHANGE 1 =======================================================
        // Split the CSV rows into one group per loop, delimited by BLANK ROWS.
        // A row is a separator if it is empty or its first cell isn't a number
        // (covers blank rows, the "x_m,y_m,z_m" header, and comment lines).
        // Numeric rows are points and accumulate into the current loop.
        var groups = [];
        var groupRows = [];
        for (var ii = 0; ii < nRows; ii += 1)
        {
            var row = pointData[ii];
            if (size(row) < 3 || !(row[cx] is number))
            {
                if (size(groupRows) > 0)
                {
                    groups = append(groups, groupRows);
                    groupRows = [];
                }
                continue;
            }
            groupRows = append(groupRows, row);
        }
        if (size(groupRows) > 0)
        {
            groups = append(groups, groupRows);
        }
        if (definition.debug)
        {
            println("Found " ~ size(groups) ~ " loop(s)");
        }
        // ======================================================================

        // ===== CHANGE 2: one spline per loop group ============================
        for (var gg = 0; gg < size(groups); gg += 1)
        {
            var loopRows = groups[gg];
            var nLoopRows = size(loopRows);

            if (definition.debug)
            {
                println("Loop " ~ gg ~ ": " ~ nLoopRows ~ " points");
            }

            // ---- Read points for this loop (skip duplicates) ----
            var rawPoints = [];
            var lastRawPoint;
            var bfirstRawPoint is boolean = true;

            for (var ii = 0; ii <= (nLoopRows - 1); ii += 1)
            {
                var row = loopRows[ii];
                var point = vector(row[cx], row[cy], row[cz]) * meter;

                if (bfirstRawPoint == false && norm(point - lastRawPoint) < 0.0001 * millimeter)
                {
                    continue;
                }

                if (definition.previewRaw == true)
                {
                    debug(context, point, DebugColor.BLUE);
                }

                rawPoints = append(rawPoints, point);
                lastRawPoint = point;
                bfirstRawPoint = false;
            }

            // ---- Sparsing algorithm (conditional) -- unchanged, per loop ----
            var splinePoints = [];

            if (definition.enableSparse)
            {
                var lastPoint;
                var lastVector;
                var lastZeroAnglePoint;
                var angle;
                var dist;
                var currentVector;

                var bfirstPoint is boolean = true;
                var bfirstNewPoint is boolean = true;
                var bskipAngle is boolean = false;
                var bskipMinDist is boolean = false;
                var bskipMaxDist is boolean = false;

                for (var jj = 0; jj <= (size(rawPoints) - 1); jj += 1)
                {
                    var point = rawPoints[jj];

                    if (bfirstPoint == false)
                    {
                        // Minimum distance
                        dist = norm(point - lastPoint);

                        if (dist < definition.minDist)
                        {
                            bskipMinDist = true;
                        } else
                        {
                            bskipMinDist = false;
                        }

                        // Angle
                        if (bfirstNewPoint == true) // Initialize orientation
                        {
                            bskipAngle = false;
                            lastVector = point - lastPoint;
                            lastVector = lastVector / norm(lastVector);
                            bfirstNewPoint = false;
                        }
                        else
                        {
                            currentVector = point - lastPoint;
                            angle = angleBetween(lastVector, currentVector / norm(currentVector));

                            if (angle < 0.0001 * degree) // zero deviation (straight line)
                            {
                                lastZeroAnglePoint = point;
                                bskipMaxDist = true;
                                bskipAngle = true;
                            } else if (angle < definition.minAngle)
                            {
                                bskipAngle = true;
                                if (dist > definition.maxDist)
                                {
                                    bskipMaxDist = false;
                                }
                            } else // angle larger than minAngle
                            {
                                bskipAngle = false;
                            }

                            bskipMinDist = bskipMinDist && (angle < 3 * definition.minAngle);
                        }

                        if ( ( bskipMinDist ) || ( bskipAngle && bskipMaxDist) )
                        {
                            continue;
                        }
                    }

                    if (definition.preview == true)
                    {
                        debug(context, point, DebugColor.RED);
                    }

                    // save previous point if large distance
                    if ( bskipAngle && (dist > definition.maxDist) )
                    {
                        splinePoints = append(splinePoints, lastZeroAnglePoint);
                    }

                    splinePoints = append(splinePoints, point);

                    bfirstPoint = false;
                    bfirstNewPoint = true;
                    lastPoint = point;

                    bskipAngle = false;
                    bskipMinDist = false;
                    bskipMaxDist = false;
                }
            }
            else
            {
                // No sparsing — use all raw points directly
                splinePoints = rawPoints;

                if (definition.preview == true)
                {
                    for (var kk = 0; kk <= (size(splinePoints) - 1); kk += 1)
                    {
                        debug(context, splinePoints[kk], DebugColor.RED);
                    }
                }
            }

            // ===== CHANGE 3: unique id per loop so all splines are created =====
            if (size(splinePoints) > 1)
            {
                opFitSpline(context, id + ("fitSpline" ~ gg), { "points" : splinePoints } );
            }
        }
        // ======================================================================
    });
