/**
 * 
 */
package com.konakart.apiexamples;

import java.util.Iterator;
import java.util.List;

import com.workingdogs.village.Record;

/**
 * An example of how to retrieve data from the KonaKart database using a custom query and the
 * KonaKart persistence layer.
 */
public class RunCustomQuery extends BaseApiExample
{
    private static final String usage = "Usage: RunCustomQuery\n" + COMMON_USAGE;

    /**
     * @param args
     */
    public static void main(String[] args)
    {
        parseArgs(args, usage, 0);

        try
        {
            /*
             * Get an instance of the KonaKart engine and login. The method called can be
             * found in BaseApiExample.java
             */
            init();

            /*
             * Run a query that selects the product id and product model from all products that have
             * an id less than 10. All Select queries need to be run using the BasePeer.executeQuery
             * command
             */
            List<Record> records = 
                    executeQuery("select products_id, products_model, products_price from products where products_id < 10");

            /*
             * Loop through the result set and print out the results. Note that the first attribute
             * in the Record object is at index = 1 and not index = 0
             */
            if (records != null)
            {
                for (Iterator<Record> iterator = records.iterator(); iterator.hasNext();)
                {
                    Record rec = iterator.next();
                    System.out.println("id = " + rec.getValue(1).asInt() + "; model = "
                            + rec.getValue(2).asString() + "; price = "
                            + rec.getValue(3).asBigDecimal(/* scale */2));
                }
            }

            /*
             * Now let's run another query to change the model name of the product with id = 1. All
             * non select queries need to be run using the BasePeer.executeStatement command
             */
            executeStatement("update products set products_model='Super Turbo' where products_id=1");

            /*
             * Read back the record that has been modified
             */
            records = executeQuery("select products_id, products_model, products_price from products where products_id = 1");

            /*
             * Print out the result. Note that the first attribute in the Record object is at index =
             * 1 and not index = 0
             */
            if (records != null && records.size() == 1)
            {
                Record rec = records.get(0);
                System.out.println("id = " + rec.getValue(1).asInt() + "; model = "
                        + rec.getValue(2).asString() + "; price = "
                        + rec.getValue(3).asBigDecimal(/* scale */2));
            }

            /*
             * Run a query that selects the orders_id of the last order processed.
             */
            records = executeQuery("SELECT max(orders_id) FROM orders");

            if (records == null)
            {
                System.out.println("No Orders Found");
            } else
            {
                System.out.println("Last Order Processed was OrderId "
                        + records.get(0).getValue(1).asString());
            }

            System.out.println("Completed Successfully");
            
        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
