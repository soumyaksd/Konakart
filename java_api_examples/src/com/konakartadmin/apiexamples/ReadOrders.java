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

import com.konakartadmin.app.AdminOption;
import com.konakartadmin.app.AdminOrder;
import com.konakartadmin.app.AdminOrderProduct;
import com.konakartadmin.app.AdminOrderSearch;
import com.konakartadmin.app.AdminOrderSearchResult;

/**
 * This class shows how to call the KonaKart Admin API to read orders from the database. Before
 * running you may have to edit BaseApiExample.java to change the username and password used to log
 * into the engine. The default values are admin@konakart.com / princess .
 */
public class ReadOrders extends BaseApiExample
{
    private static final String usage = "Usage: ReadOrders\n" + COMMON_USAGE;

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

            AdminOrderSearch search = new AdminOrderSearch();
            AdminOrderSearchResult orderResult = eng.getOrders(sessionId, search, 0, 100, -1);

            AdminOrder[] orders = orderResult.getOrders();

            for (int i = 0; i < orders.length; i++)
            {
                AdminOrder tmpOrder = orders[i];
                int orderId = tmpOrder.getId();

                AdminOrder order = eng.getOrderForOrderId(sessionId, orderId);

                AdminOrderProduct[] products = order.getOrderProducts();
                for (int j = 0; j < products.length; j++)
                {
                    AdminOrderProduct orderProduct = products[j];

                    AdminOption[] options = orderProduct.getOpts();

                    if (options != null)
                    {
                        System.out.println("\n\nOrder Id:     " + order.getId() + "\n");
                        System.out.println(orderProduct.toString());

                        for (int k = 0; k < options.length; k++)
                        {
                            AdminOption option = options[k];

                            System.out.println("Option " + k + ") [" + option.getId() + "] "
                                    + option.getName() + " " + option.getValue());
                        }
                    }
                }
            }
        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
