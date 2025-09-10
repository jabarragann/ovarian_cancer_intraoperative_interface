import os
import slicer
import vtk

def get_visible_segments(node_name: str)->list[str]:
    all_visible_segments = []

    segNode = slicer.util.getNode(node_name)  # or getNodesByClass()[0]

    segmentation = segNode.GetSegmentation()
    displayNode = segNode.GetDisplayNode()

    segmentIDs = vtk.vtkStringArray()
    segmentation.GetSegmentIDs(segmentIDs)

    for i in range(segmentIDs.GetNumberOfValues()):
        segmentID = segmentIDs.GetValue(i)
        segmentName = segmentation.GetSegment(segmentID).GetName()

        # Check visibility
        if displayNode.GetSegmentVisibility(segmentID):
            all_visible_segments.append(segmentName)

    return all_visible_segments

def display_visible_segments():
    node_name = "total_segmentation"
    visible_segments = get_visible_segments(node_name)

    print("Visible segments:")
    for segment in visible_segments:
        print(f" - {segment}")

def export_to_mesh():
    nodes_of_interest = ["total_segmentation"]

    ## Hardcoded names
    # # segmentNamesToExport = ["liver", "gallbladder", "inferior vena cava", "portal vein and splenic vein"]
    # segmentNamesToExport = [ "spleen", "gallbladder", "liver", "stomach", "inferior lobe of left lung", "inferior lobe of right lung",
    #     "heart", "aorta", "inferior vena cava", "portal vein and splenic vein" ]
    ## Visible names
    segmentNamesToExport = get_visible_segments("total_segmentation")

    outputGlbPath = "/home/juan95/research/3dreconstruction/slicer_scripts/output/"

    # Create a parent model folder
    modelFolder = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelHierarchyNode", "ExportedSegmentsFolder")
    segmentationNodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")

    for segNode in segmentationNodes:
        if segNode.GetName() not in nodes_of_interest:
            continue

        segmentation = segNode.GetSegmentation()
        segmentIDs = vtk.vtkStringArray()
        segmentation.GetSegmentIDs(segmentIDs)

        print(f"Processing Segmentation Node: {segNode.GetName()}")

        for i in range(segmentIDs.GetNumberOfValues()):
            segmentID = segmentIDs.GetValue(i)
            segment = segmentation.GetSegment(segmentID)
            segmentName = segment.GetName()

            if segmentName in segmentNamesToExport:
                print(f"Exporting segment: {segmentName}")

                # Export this segment into the model node (requires vtkStringArray)
                segmentIDArray = vtk.vtkStringArray()
                segmentIDArray.InsertNextValue(segmentID)
                slicer.modules.segmentations.logic().ExportSegmentsToModels(segNode, segmentIDArray, True)

                # Find the newly created model node (by name)
                modelNode = slicer.util.getNode(f"{segmentName}")
                # Save model as OBJ
                slicer.util.saveNode(modelNode, outputGlbPath+f"{segmentName}.obj")
                # Erase model
                slicer.mrmlScene.RemoveNode(modelNode)

if __name__ == "__main__":
    ### uncomment the needed method before uploading to 3d Slicer with ctrl + g

    export_to_mesh()
    # display_visible_segments()