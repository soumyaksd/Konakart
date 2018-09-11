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

import com.konakart.util.KKConstants;
import com.konakart.util.Utils;
import com.konakartadmin.app.AdminCategory;

/**
 * This class shows how to call the KonaKart Admin API to read categories from the database. Before
 * running you may have to edit BaseApiExample.java to change the username and password used to log
 * into the engine. The default values are admin@konakart.com / princess .
 */
public class ReadCategories extends BaseApiExample
{
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
            parseArgs(args, COMMON_USAGE, 0);

            /*
             * Get an instance of the Admin KonaKart engine and login. The method called can be
             * found in BaseApiExample.java
             */
            init(getEngineMode(), getStoreId(), getEngClassName(), isCustomersShared(),
                    isProductsShared(), isCategoriesShared(), null);

            /*
             * Read the store name
             */
            String storeName = getEng().getConfigurationByKey(getSessionId(), "STORE_NAME")
                    .getConfigurationValue();

            /*
             * Read the category tree
             */
            AdminCategory[] catTree = getEng().getCategoryTree(KKConstants.DEFAULT_LANGUAGE_ID,
                    true);

            /*
             * Print out each Category's name and then all its child categories
             */
            System.out.println("Category Tree for " + storeName);
            System.out.println(Utils.makeString(storeName.length() + 18, '-'));

            for (int i = 0; i < catTree.length; i++)
            {
                printCategoryTree(catTree[i], 1);
            }
        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    private static void printCategoryTree(AdminCategory cat, int catLevel)
    {
        System.out.println(pad(catLevel) + leftPad(cat.getId(), 4) + ") " + cat.getName() + " ("
                + +cat.getNumberOfProducts() + ")");

        if (cat.getChildren() == null || cat.getChildren().length == 0)
        {
            return;
        }

        for (int c = 0; c < cat.getChildren().length; c++)
        {
            printCategoryTree(cat.getChildren()[c], catLevel + 1);
        }

        System.out.println(pad(catLevel) + "---------------------------------");
    }

    private static String leftPad(int i, int n)
    {
        return Utils.padLeft(i, n);
    }

    private static String pad(int n)
    {
        return Utils.makeString(n * 3, ' ');
    }
}
