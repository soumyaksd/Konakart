package com.infy;

public class Outerclass {
   // instance method of the outer class 
	int outer =1;
	public Outerclass(){
		
	}
	void my_Method1() {
      int method1 = 23;

      // method-local inner class
      class MethodInner_Demo {
         public void print() {
            System.out.println("This is method inner class ");	   
         }   
      } // end of inner class
	   
      // Accessing the inner class
      MethodInner_Demo inner = new MethodInner_Demo();
      inner.print();
   }
   void my_Method2() {
	      int method2 = 23;
	      System.out.println(method2);
	}
   // inner class
   public class Inner_Demo {
	   public Inner_Demo(){
			
		}
      public void print() {
         System.out.println("This is the inner class");
      }
   }
}

public class Mainclass {
	public static void main(String args[]) {
	      Outerclass main = new Outerclass();
	      main.my_Method1();	   	   
	   }
}