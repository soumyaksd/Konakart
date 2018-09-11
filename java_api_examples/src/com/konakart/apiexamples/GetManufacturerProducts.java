//
// (c) 2006 DS Data Systems UK Ltd, All rights reserved.
//
// DS Data Systems and KonaKart and their respective logos, are 
// trademarks of DS Data Systems UK Ltd. All rights reserved.
//
// The information in this document is free software; you can redistribute 
// it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This software is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
package com.konakart.apiexamples;

import com.konakart.app.DataDescConstants;
import com.konakart.app.DataDescriptor;
import com.konakart.app.ProductSearch;
import com.konakart.appif.ManufacturerIf;
import com.konakart.appif.ProductIf;
import com.konakart.appif.ProductsIf;

/**
 * This class shows how to call the KonaKart API to retrieve manufacturer's Products.
 * 
 * Before running you may have to edit BaseApiExample.java to change the username and password used
 * to log into the engine. The default values are doe@konakart.com / password .
 */
public class GetManufacturerProducts extends BaseApiExample
{
    private static final String usage = "Usage: GetManufacturerProducts\n" + COMMON_USAGE;

    /**
     * @param args
     */
    public static void main(String[] args)
    {
        parseArgs(args, usage, 0);

        try
        {
            /*
             * Get an instance of the KonaKart engine and login. The method called can be found in
             * BaseApiExample.java
             */
            init();

            // First get all the Manufacturers...
            ManufacturerIf manufacturer[] = eng.getAllManufacturers();

            System.out.println("Total number of Manufacturers: " + manufacturer.length);

            DataDescriptor datadesc = new DataDescriptor();
            datadesc.setOffset(0);
            datadesc.setLimit(DataDescConstants.MAX_ROWS);

            for (int man = 0; man < manufacturer.length; man++)
            {
                System.out.println("Manufacturer Name: " + manufacturer[man].getName());

                // Now for each Manufacturer, we'll get all their first MAX_ROWS products...
                
                ProductSearch prodSearch = new ProductSearch();
                prodSearch.setManufacturerId(manufacturer[man].getId());

                ProductsIf products = eng.searchForProducts(sessionId, datadesc, prodSearch, -1);
                ProductIf[] product_list = products.getProductArray();
                
                for (int j = 0; j < product_list.length; j++)
                {
                    System.out.println("The name of product is " + product_list[j].getName());
                }
            }
        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
