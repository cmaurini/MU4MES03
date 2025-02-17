# Exercice 

## TD 3 and Homework 2: Frame modeling 

- Work in group of 4 students
- Only one report for the group must be submitted. The report will be composed of two documents
    1. A detailed report with the calculations
    2. A presentation of the work done with
        - One or two slides with the analytical solution with the result for the displacement of the node D, idea of the method of solution and diagram for the bending moment in the bars.
        - One slide with the numerical solution.
        - One slide with the experiment.
        - One slide of comparisons/conclusions.
    Name the file as `DM_frame_report_nom1_nom2_nom3_nom4.pdf` and `DM_frame_presentation_nom1_nom2_nom3_nom4.pdf`
- Submit on moodle at https://moodle-sciences-24.sorbonne-universite.fr/mod/assign/view.php?id=163171
- Specify at the end of the file the contribution of each member of the group (e.g. student 1 worked on the analytical part, student 2 and 3 worked on the code, student 4 on  the experiments, etc.). Obviously, all the members of the group are responsible for the work of the group.
- You will do a 5 minutes presentation of the work done in class on March 3rd.

### I. Mola beam (TD2 Exercice 2)

Resume your results on the Exercice 2 of TD2 on the comparison between the analytical solution and the experimental finding for the cantilever `Mola` beam. You should give the following results and comment
- The value of the equivalent bending stiffness $(EI)_\mathrm{equiv}$ of the beam
- The analytical solution for the displacement of the tip of the beam
- The experimental value of the displacement of the tip of the beam

### II. Frame structure

We consider the frame in Figure. 

```{figure} frame.png
---
width: 70%
name: Frame Structure
---
Frame structure under load
```

####  Elementary tasks

We split the work in the following four tasks, that can be performed in parallel by the group members:

1. Reproduce the structure with the Mola model and take a picture the deformation of the structure under the imposed load.
2. Define the corresponding frame model by modifying the example in the notebook.
3. Solve the problem analytically using the method of the minimum of the complementary potential energy and compute the displacement of the node $D$. Plot the diagram of the normal force, shear force and bending moments in the bars.

#### Quantitative Verification of the numerical model

Compare the results of the analytical solution with the numerical solution obtained with the code and comment.

#### Qualitative validation of the model

Perform a qualitative comparison of the deformed configuration:
    - Take a photo of the deformed configuration of the real structure under the imposed load
    - Compare qualitatively with the deformed configuration plot obtained with your code