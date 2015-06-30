static char help[] = "Reads in a matrix in ASCII MATLAB format (I,J,A), read in vectors rhs and exact_solu in ASCII format.\n\
    Writes them using the PETSc sparse format.\n\
    Note: I and J start at 1, not 0, n\
     Input parameters are:\n\
     -Ain  <filename> : input matrix in ascii format\n\
     -rhs  <filename> : input rhs in ascii format\n\
     -solu  <filename> : input true solution in ascii format\n\\n";
   
     /*
     Example: ./ex2 -Ain Ain -rhs rhs -solu solu
     with the datafiles in the followig format:
     Ain (I and J start at 0):
     ------------------------
     3 3 6
     0 0 1.0
     0 1 2.0
     1 0 3.0
     1 1 4.0
     1 2 5.0
     2 2 6.0
   
     rhs
     ---
     0 3.0
     1 12.0
     2 6.0
   
     solu
     ----
     0 1.0
     0 1.0
     0 1.0
     */
   
     #include <petscmat.h>
     #include <petscksp.h>
   
     #undef __FUNCT__
     #define __FUNCT__ "main"
     int main(int argc,char **args)
     {
       Mat            A,P;			/* A is our main linear system matrix and P our preconditioner*/
       Vec            b,u,u_tmp,u_cg,x;
       char           Ain[PETSC_MAX_PATH_LEN], rhs[PETSC_MAX_PATH_LEN], solu[PETSC_MAX_PATH_LEN]; 
       PetscErrorCode ierr;
       int            m,n,nz,dummy; /* these are fscaned so kept as int */
       PetscInt       i,col,row,shift = 1,sizes[3],nsizes,its;
       PetscScalar    val,temp=1,rnd_tmp;
       PetscReal      res_norm,norm;
       FILE           *Afile,*bfile,*ufile;
       PetscViewer    view;
       PetscBool      flg_A,flg_b,flg_u,flg;
       PetscMPIInt    size;
       PetscRandom	 rnd;			/* For creating our random initial guess */
   
       KSP            ksp;			/* linear solver context */
       KSPConvergedReason reason;    /* The reason our solver converged */
   
       PetscInitialize(&argc,&args,(char *)0,help);
       ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
       if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");
   
       /* Read in matrix, rhs and exact solution from ascii files */
   
       ierr = PetscOptionsGetString(PETSC_NULL,"-Ain",Ain,PETSC_MAX_PATH_LEN,&flg_A);CHKERRQ(ierr);
       ierr = PetscOptionsHasName(PETSC_NULL,"-noshift",&flg);CHKERRQ(ierr);
       if (flg) shift = 0;
       if (flg_A){
         ierr = PetscPrintf(PETSC_COMM_SELF,"\n Read matrix in ascii format ...\n");CHKERRQ(ierr);
         ierr = PetscFOpen(PETSC_COMM_SELF,Ain,"r",&Afile);CHKERRQ(ierr); 
         nsizes = 3;
         ierr = PetscOptionsGetIntArray(PETSC_NULL,"-nosizesinfile",sizes,&nsizes,&flg);CHKERRQ(ierr);
         if (flg) {
           if (nsizes != 3) SETERRQ(PETSC_COMM_WORLD,1,"Must pass in three m,n,nz as arguments for -nosizesinfile");
           m = sizes[0];
           n = sizes[1];
           nz = sizes[2];
         } else {
           fscanf(Afile,"%d %d %d\n",&m,&n,&nz);
         }
         ierr = PetscPrintf(PETSC_COMM_SELF,"m: %d, n: %d, nz: %d \n", m,n,nz);CHKERRQ(ierr);
         if (m != n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Number of rows, cols must be same...\n");
         ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
         ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
         ierr = MatSetFromOptions(A);CHKERRQ(ierr);
         ierr = MatSeqAIJSetPreallocation(A,nz,PETSC_NULL);CHKERRQ(ierr);
     
         for (i=0; i<nz; i++) {
           fscanf(Afile,"%d %d %le\n",&row,&col,&val);
           ierr = MatSetValues(A,1,&row,1,&col,&val,INSERT_VALUES);CHKERRQ(ierr);
         }
         ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
         ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
         fflush(stdout);
         fclose(Afile);  
   	
     	// Print the asssembled Matrix A with MatView()
   		ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrix created in the above lines:\n");CHKERRQ(ierr);
     	ierr = MatView(A,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
   	
     	/*
     	// Starting to initialize the preconditioner matrix P
     	ierr = PetscPrintf(PETSC_COMM_SELF,"Creating our I matrix to   be used as preconditioner...");CHKERRQ(ierr);
     	ierr = MatCreate(PETSC_COMM_SELF,&P);CHKERRQ(ierr);
     	ierr = MatSetSizes(P,PETSC_DECIDE,PETSC_DECIDE,n,n); CHKERRQ(ierr);
     	ierr = MatSetFromOptions(P);CHKERRQ(ierr);
     	ierr = MatSeqAIJSetPreallocation(P,n,PETSC_NULL);CHKERRQ(ierr);
   	
   
     	for (i=0; i<n; i++) {
   			ierr = MatSetValues(P,1,&i,1,&i,&temp,INSERT_VALUES);
     	}
     	ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
     	ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
     	fflush(stdout);
   
     	PetscPrintf(PETSC_COMM_WORLD,"Matrix created as preconditioner...:\n");
     	MatView(P,PETSC_VIEWER_STDOUT_SELF);
     }
     // End of matrix's P initialization and printing 
     */
   
     // Starting to initialize the vector b
     ierr = PetscOptionsGetString(PETSC_NULL,"-rhs",rhs,PETSC_MAX_PATH_LEN,&flg_b);CHKERRQ(ierr);
     if (flg_b){
       ierr = VecCreate(PETSC_COMM_SELF,&b);CHKERRQ(ierr);
       ierr = VecSetSizes(b,PETSC_DECIDE,n);CHKERRQ(ierr);
       ierr = VecSetFromOptions(b);CHKERRQ(ierr);
       ierr = PetscPrintf(PETSC_COMM_SELF,"\n Read rhs in ascii format ...\n");CHKERRQ(ierr);
       ierr = PetscFOpen(PETSC_COMM_SELF,rhs,"r",&bfile); CHKERRQ(ierr); 
       for (i=0; i<n; i++) {      
         fscanf(bfile,"%d %le\n",&dummy,(double*)&val); 
         ierr = VecSetValues(b,1,&i,&val,INSERT_VALUES); CHKERRQ(ierr);
       }
       ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
       ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
       fflush(stdout);
       fclose(bfile);
     }
     // End of vector's b initialization
     // Printing the vector b for debugging
     ierr = PetscPrintf(PETSC_COMM_WORLD, "Vector b created in the above lines...\n");CHKERRQ(ierr);
     ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
   
     // Starting to initialize the solution vector u
     ierr = PetscOptionsGetString(PETSC_NULL,"-solu",solu,PETSC_MAX_PATH_LEN,&flg_u);CHKERRQ(ierr);
     if (flg_u){
       ierr = VecCreate(PETSC_COMM_SELF,&u);CHKERRQ(ierr);
       ierr = VecSetSizes(u,PETSC_DECIDE,n);CHKERRQ(ierr);
       ierr = VecSetFromOptions(u);CHKERRQ(ierr);
       ierr = PetscPrintf(PETSC_COMM_SELF,"\n Read exact solution in ascii format ...\n");CHKERRQ(ierr);
       ierr = PetscFOpen(PETSC_COMM_SELF,solu,"r",&ufile); CHKERRQ(ierr); 
       for (i=0; i<n; i++) {
         fscanf(ufile,"%d  %le\n",&dummy,(double*)&val);
         ierr = VecSetValues(u,1,&i,&val,INSERT_VALUES); CHKERRQ(ierr);
       }
       ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
       ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
       fflush(stdout);
       fclose(ufile);
     }
     // End of vector's u initialization
     // Printing the exact solution vector u
     PetscPrintf(PETSC_COMM_WORLD, "Vector u created in the above lines...\n");
     VecView(u,PETSC_VIEWER_STDOUT_WORLD);
     /*
     // Creating our random initial guess using u
     VecDuplicate(u,&u_cg);
     PetscRandomCreate(PETSC_COMM_WORLD, &rnd);
     PetscRandomSetFromOptions(rnd);
     VecSetRandom(u_cg,rnd);
     PetscRandomDestroy(&rnd);
     PetscPrintf(PETSC_COMM_WORLD, "Random vector with elements in [-1,1] u_cg created in the above lines...\n");
     VecView(u_cg, PETSC_VIEWER_STDOUT_WORLD);
     PetscPrintf(PETSC_COMM_WORLD, "Random vector multiplied by 0,1 created in the above lines...\n");
     VecScale(u_cg, 0.1);
     // Calculating mean value of vector u elements
     VecSum(u,&rnd_tmp);
     rnd_tmp=rnd_tmp/n;
     PetscPrintf(PETSC_COMM_WORLD, "The mean value of u calcilated: %le\n",rnd_tmp);
     // And scaling again our u_cg with the mean value
     VecScale(u_cg, rnd_tmp);
     // u_cg contains the random vector
     PetscPrintf(PETSC_COMM_WORLD,"Random vector u_cg created...\n");
     VecView(u_cg, PETSC_VIEWER_STDOUT_WORLD);
     // Contruction of the random initial guess as the vector sum of u and u_cg
     VecAXPY(u_cg,1.0,u);
     PetscPrintf(PETSC_COMM_WORLD,"Our final initial guess vector u_cg created...\n");
     VecView(u_cg, PETSC_VIEWER_STDOUT_WORLD);
     // End of creating of our random initial guess u_cg
     */
   
     /* Check accuracy of the data */
     if (flg_A & flg_b & flg_u){
       ierr = VecDuplicate(u,&u_tmp);CHKERRQ(ierr); 
       ierr = MatMult(A,u,u_tmp);CHKERRQ(ierr);
       ierr = VecAXPY(u_tmp,-1.0,b);CHKERRQ(ierr);
       ierr = VecNorm(u_tmp,NORM_2,&res_norm);CHKERRQ(ierr);
       ierr = PetscPrintf(PETSC_COMM_SELF,"\n Accuracy of the reading data: | b - A*u |_2 : %g \n",res_norm);CHKERRQ(ierr);
       ierr = VecDestroy(&u_tmp);CHKERRQ(ierr);
     }
   
     /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                   Create the linear solver and set various options
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
     /*
        Create linear solver context
     */
   
     ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
   
     /*
        Set operators. Here the matrix that defines the linear system
        also serves as the preconditioning matrix.
     */
   
     ierr = KSPSetOperators(ksp,A,Á);CHKERRQ(ierr);
   
     /*
        Set linear solver defaults for this problem (optional).
        - By extracting the KSP and PC contexts from the KSP context,
          we can then directly call any KSP and PC routines to set
          various options.
        - The following two statements are optional; all of these
          parameters could alternatively be specified at runtime via
          KSPSetFromOptions().  All of these defaults can be
          overridden at runtime, as indicated below.
     */
     ierr = KSPSetTolerances(ksp,PETSC_DEFAULT ,PETSC_DEFAULT ,PETSC_DEFAULT, PETSC_DEFAULT);CHKERRQ(ierr);
     /*
       Set runtime options, e.g.,
           -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
       These options will override those specified above as long as
       KSPSetFromOptions() is called _after_ any other customization
       routines.
     */
     ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
   
     /*
   	Creating the solution vector of x for saving the iterative's method
   	solution
     */
     ierr = VecCreate(PETSC_COMM_SELF,&x);CHKERRQ(ierr);
     ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
     ierr = VecSetFromOptions(x);CHKERRQ(ierr);
     ierr = VecCopy(u_cg,x);CHKERRQ(ierr);					//setting the solution vector of the iterative method equal to random_vec + exact_solution (u_cg)
     //ierr = VecCopy(u,x);CHKERRQ(ierr);
     //ierr = VecScale(x,1);CHKERRQ(ierr);
     ierr = PetscPrintf(PETSC_COMM_WORLD,"\nThe initial guesss X:\n");CHKERRQ(ierr);
     ierr = VecView(x,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
   
     /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                         Solve the linear system
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
     /*
     KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);
     */
     // Setting default solver the CG method
     ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
     
     /*
     // Setting default solver the GMRES method
     KSPSetType(ksp,KSPGMRES);CHKERRQ(ierr);
     */
   
     ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
     // Checking the divergence reason
     ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
   
     if(reason==KSP_DIVERGED_INDEFINITE_PC) {
   	 ierr = PetscPrintf(PETSC_COMM_WORLD,"\nDivergence because of indefinite preconditioner;\n");CHKERRQ(ierr);
     }
     else if (reason<0) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\nOther kind of divergence: this should not happen.\n");CHKERRQ(ierr);
     } 
     else {
        ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
     }
   
      /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                         Check solution and clean up
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   
     // Print the solution-approximation vector
     PetscPrintf(PETSC_COMM_WORLD,"\nThe approximation of the solution x:\n");
     VecView(x,PETSC_VIEWER_STDOUT_SELF);
     
     // Check the error
     ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
     ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
     ierr = PetscPrintf(PETSC_COMM_WORLD,"Vector error:\n");CHKERRQ(ierr);
     ierr = VecView(x,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
     ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
   
     /*
        Print convergence information.  PetscPrintf() produces a single
        print statement from all processes that share a communicator.
        An alternative is PetscFPrintf(), which prints to a file.
     */
     ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %e iterations %D\n",norm,its);CHKERRQ(ierr);
   
   
     ierr = MatDestroy(&A);CHKERRQ(ierr);
     if (flg_b) {ierr = VecDestroy(&b);CHKERRQ(ierr);}
     if (flg_u) {ierr = VecDestroy(&u);CHKERRQ(ierr);}
     ierr = VecDestroy(&x);CHKERRQ(ierr);
     ierr = VecDestroy(&b);CHKERRQ(ierr);
     ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
     ierr = PetscFinalize();
     return 0;}

