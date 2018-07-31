*tristars*: Match coordinate lists using similar triangles.

Demo
~~~~

Generate a mock catalog with 50 sources in a parent catalog and 10 sources in a transformed subset of this catalog.

   >>> from tristars import test
   >>> test.test(N1=50, N2=10, err=1, tr=[40,50], rot=0.1)
   
   .. image:: docs/_static/tristars_test.png
   