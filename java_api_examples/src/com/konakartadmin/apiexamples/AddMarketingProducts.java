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

import java.util.HashMap;

import com.konakart.app.DataDescConstants;
import com.konakart.util.KKConstants;
import com.konakartadmin.app.AdminDataDescriptor;
import com.konakartadmin.app.AdminProduct;
import com.konakartadmin.app.AdminProductSearch;
import com.konakartadmin.app.AdminProducts;
import com.konakartadmin.app.KKAdminException;
import com.konakartadmin.bl.KonakartAdminConstants;

/**
 * This class shows how to call the KonaKart Admin API to insert some marketing products for each
 * product in the database. This is useful for setting up a demo database. It adds marketing
 * products from the same category as the selected product. Before running you may have to edit
 * BaseApiExample.java to change the username and password used to log into the engine. The default
 * values are admin@konakart.com / princess .
 */
public class AddMarketingProducts extends BaseApiExample
{
    private static final String usage = "Usage: AddMarketingProducts\n" + COMMON_USAGE;

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

            /*
             * Create a hash map to store products for a category
             */
            HashMap<Integer, AdminProduct[]> prodMap = new HashMap<Integer, AdminProduct[]>();
            /*
             * Loop through all of the products
             */
            AdminProductSearch prodSearch = new AdminProductSearch();
            AdminDataDescriptor dataDesc = new AdminDataDescriptor(
                    DataDescConstants.ORDER_BY_DATE_ADDED, 0, 100);

            AdminProducts prods = eng.searchForProducts(sessionId, dataDesc, prodSearch,
                    KKConstants.DEFAULT_LANGUAGE_ID);
            if (prods.getProductArray() != null)
            {
                for (int i = 0; i < prods.getProductArray().length; i++)
                {
                    AdminProduct prod = prods.getProductArray()[i];
                    prod = eng.getProduct(sessionId, prod.getId());
                    // Get the products in same category
                    int catId = prod.getCategoryId();
                    AdminProduct[] catProds = prodMap.get(new Integer(catId));
                    if (catProds == null)
                    {
                        catProds = getProdsForCat(catId);
                        prodMap.put(new Integer(catId), catProds);
                    }
                    if (catProds != null && catProds.length > 1)
                    {
                        AdminProduct[] mktProds = new AdminProduct[catProds.length - 1];
                        int k = 0;
                        for (int j = 0; j < catProds.length; j++)
                        {
                            AdminProduct mktProd = catProds[j];
                            if (mktProd.getId() != prod.getId())
                            {
                                mktProds[k++] = mktProd;
                            }
                        }
                        eng.addRelatedProducts(sessionId, mktProds, prod.getId(),
                                KonakartAdminConstants.CROSS_SELL_RELATION);
                    }
                }
            }

        } catch (Exception e)
        {
            e.printStackTrace();
        }

    }

    /**
     * 
     * @param categoryId
     * @return Return the products for categoryId
     * @throws KKAdminException
     */
    static private AdminProduct[] getProdsForCat(int categoryId) throws KKAdminException
    {
        AdminProductSearch prodSearch = new AdminProductSearch();
        prodSearch.setCategoryId(categoryId);
        AdminDataDescriptor dataDesc = new AdminDataDescriptor(
                DataDescConstants.ORDER_BY_DATE_ADDED, 0, 100);

        AdminProducts prods = eng.searchForProducts(sessionId, dataDesc, prodSearch,
                KKConstants.DEFAULT_LANGUAGE_ID);
        return prods.getProductArray();
    }

}
