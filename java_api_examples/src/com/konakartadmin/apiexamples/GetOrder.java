//
// (c) 2004-2015 DS Data Systems UK Ltd, All rights reserved.
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
package com.konakartadmin.apiexamples;

import com.konakartadmin.app.AdminOrder;

/**
 * This class shows how to call the KonaKart Admin API to retrieve an order from the database. Before
 * running you may have to edit BaseApiExample.java to change the username and password used to log
 * into the engine. The default values are admin@konakart.com / princess .
 */
public class GetOrder extends BaseApiExample
{
    private static final String usage = "Usage: GetOrder\n" + COMMON_USAGE;

    /**
     * @param args
     */
    public static void main(String[] args)
    {
        try
        {
            /*
             * Parse the command line arguments
             */
            parseArgs(args, usage, 0);
            
            /*
             * Get an instance of the Admin KonaKart engine and login. The method called can be
             * found in BaseApiExample.java
             */
            init(getEngineMode(), getStoreId(), getEngClassName(), isCustomersShared(),
                    isProductsShared(), isCategoriesShared(), null);


            // We assume that an order exists with id = 1
            AdminOrder order = eng.getOrderForOrderId(sessionId, 1);
            
            if (order != null)
            {
                System.out.println(order.toStringBrief());
            }
            

        } catch (Exception e)
        {
            e.printStackTrace();
        }

    }

}
