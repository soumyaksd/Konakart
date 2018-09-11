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
import com.konakartadmin.app.AdminDataDescriptor;
import com.konakartadmin.app.AdminProduct;
import com.konakartadmin.app.AdminProductDescription;
import com.konakartadmin.app.AdminProductSearch;
import com.konakartadmin.app.AdminProducts;

/**
 * This class shows how to call the KonaKart Admin API to update all the product descriptions in the
 * database.
 * <p>
 * Before running you may have to edit BaseApiExample.java to change the username and password used
 * to log into the engine. The default values are admin@konakart.com / princess .
 */
public class UpdateProductDescriptions extends BaseApiExample
{
    private static final String usage = "Usage: UpdateProductDescriptions\n" + COMMON_USAGE;

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

            // Loop through all the products, for all languages, 10 at a time

            int changesMade = 0;
            int prodsFound = 0;
            boolean moreProducts = true;

            AdminProductSearch prodSearch = new AdminProductSearch();

            AdminDataDescriptor dataDesc = new AdminDataDescriptor();
            int currentOffset = 0;
            int groupSize = 10;
            dataDesc.setShowInactive(true);
            dataDesc.setShowInvisible(true);
            dataDesc.setLimit(groupSize);

            while (moreProducts)
            {
                dataDesc.setOffset(currentOffset);
                AdminProducts prods = eng.searchForProducts(sessionId, dataDesc, prodSearch,
                        KKConstants.DEFAULT_LANGUAGE_ID);

                if (prods.getProductArray().length < groupSize)
                {
                    moreProducts = false;
                }

                prodsFound += prods.getProductArray().length;
                
                // For each product in the group returned..

                for (int p = 0; p < prods.getProductArray().length; p++)
                {
                    AdminProduct liteProd = prods.getProductArray()[p];
                    AdminProduct heavyProd = eng.getProduct(sessionId, liteProd.getId());

                    // Now change something in the product to save

                    boolean changeMade = false;

                    for (int pd = 0; pd < heavyProd.getDescriptions().length; pd++)
                    {
                        AdminProductDescription apd = heavyProd.getDescriptions()[pd];
                        if (apd.getDescription().contains("" + ((char) 13)))
                        {
                            apd.setDescription(apd.getDescription().replace((char) 13, (char) 10));
                            System.out.println("Updated Product " + heavyProd.getId()
                                    + " language " + apd.getLanguageId());
                            changeMade = true;
                        }
                    }

                    // Save the changes if any were made
                    if (changeMade)
                    {
                        eng.editProduct(sessionId, heavyProd);
                        changesMade++;
                    }
                }

                currentOffset += groupSize;
            }

            // Logout
            eng.logout(sessionId);

            System.out.println(prodsFound + " products processed");
            System.out.println(changesMade + " changes made");
            System.out.println("UpdateProductDescriptions Completed successfully");

        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
