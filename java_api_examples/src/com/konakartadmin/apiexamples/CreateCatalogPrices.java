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

import java.math.BigDecimal;
import java.util.ArrayList;

import com.konakart.app.DataDescConstants;
import com.konakartadmin.app.AdminCatalog;
import com.konakartadmin.app.AdminCatalogRule;
import com.konakartadmin.app.AdminCatalogSearch;
import com.konakartadmin.app.AdminCatalogSearchResult;
import com.konakartadmin.app.AdminCategory;
import com.konakartadmin.app.AdminCreateCatalogPriceOptions;
import com.konakartadmin.app.AdminCreateCatalogPricesResult;
import com.konakartadmin.app.AdminDataDescriptor;
import com.konakartadmin.app.AdminProduct;
import com.konakartadmin.app.AdminProductMgrOptions;
import com.konakartadmin.app.AdminProductSearch;
import com.konakartadmin.app.AdminProducts;
import com.konakartadmin.bl.KonakartAdminConstants;

/**
 * This class shows how to add some category based rules and then to create prices for a catalog
 * based on those rules. Before running you may have to edit BaseApiExample.java to change the
 * username and password used to log into the engine. The default values are admin@konakart.com /
 * princess .
 */
public class CreateCatalogPrices extends BaseApiExample
{
    private static final String usage = "Usage: CreateCatalogPrices\n" + COMMON_USAGE;

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
             * Define the catalog name
             */
            String catalogName = "cat1";

            /*
             * See if catalog already exists. If so delete it.
             */
            AdminCatalogSearch search = new AdminCatalogSearch();
            search.setCatalogName(catalogName);
            AdminCatalogSearchResult result = eng.getCatalogs(sessionId, search, 0, 1000);
            if (result.getCatalogs() != null)
            {
                for (int i = 0; i < result.getCatalogs().length; i++)
                {
                    AdminCatalog cat = result.getCatalogs()[i];
                    eng.deleteCatalog(sessionId, cat.getId());
                }
            }

            /*
             * Insert a catalog
             */
            AdminCatalog cat1 = new AdminCatalog();
            cat1.setName(catalogName);
            cat1.setDescription("Demo Catalog");
            cat1.setUseCatalogPrices(true);
            cat1.setUseCatalogQuantities(false);
            eng.insertCatalog(sessionId, cat1);

            /*
             * Create rules to exclude all top level categories except for the first one. All
             * products in the first category are given a fixed price of 5.00
             */
            ArrayList<AdminCatalogRule> ruleList = new ArrayList<AdminCatalogRule>();
            AdminCategory[] topCats = eng.getCategoryTree(
                    KonakartAdminConstants.DEFAULT_LANGUAGE_ID, /* getNumProducts */
                    false);
            int includeCat = 0, excludeCat = 0;
            for (int i = 0; i < topCats.length; i++)
            {
                AdminCategory cat = topCats[i];
                AdminCatalogRule rule = new AdminCatalogRule();
                rule.setCategoryId(cat.getId());
                if (i == 0)
                {
                    rule.setExclude(false);
                    rule.setFixedPrice(new BigDecimal("5.00"));
                    includeCat = cat.getId();
                } else
                {
                    rule.setExclude(true);
                    excludeCat = cat.getId();
                }
                ruleList.add(rule);
            }

            /*
             * Save the rules
             */
            eng.addCatalogRules(sessionId, catalogName, ruleList.toArray(new AdminCatalogRule[0]));

            /*
             * Create the prices using the saved rules
             */
            AdminCreateCatalogPriceOptions options1 = new AdminCreateCatalogPriceOptions();
            options1.setProductFetchLimit(1000);
            AdminCreateCatalogPricesResult ret = eng.createCatalogPricesFromRules(sessionId,
                    catalogName, options1);
            System.out.println("\n== " + ret.getNumProductsInCatalog()
                    + " products inserted in the catalog " + ret.getCatalogId() + " ==");

            /*
             * Search for products in included category
             */
            AdminProductMgrOptions options = new AdminProductMgrOptions();
            options.setCatalogId(catalogName);
            options.setUseExternalPrice(true);
            AdminDataDescriptor dd = new AdminDataDescriptor(DataDescConstants.ORDER_BY_DATE_ADDED,
                    0, 100);
            AdminProductSearch prodSearch = new AdminProductSearch();
            prodSearch.setCategoryId(includeCat);
            prodSearch.setSearchCategoryTree(true);
            AdminProducts prods = eng.searchForProductsWithOptions(sessionId, dd, prodSearch,
                    KonakartAdminConstants.DEFAULT_LANGUAGE_ID, options);

            System.out.println("\n== Fixed Prices in Category id = " + includeCat + " ==");
            if (prods != null && prods.getProductArray() != null)
            {
                for (int i = 0; i < prods.getProductArray().length; i++)
                {
                    AdminProduct prod = prods.getProductArray()[i];
                    System.out.println("Price for prod = " + prod.getName() + " = "
                            + prod.getPriceExTax());
                }
            }

            /*
             * Search for products in excluded category
             */
            prodSearch.setCategoryId(excludeCat);
            prodSearch.setSearchCategoryTree(true);
            prods = eng.searchForProductsWithOptions(sessionId, dd, prodSearch,
                    KonakartAdminConstants.DEFAULT_LANGUAGE_ID, options);
            if (prods != null && prods.getTotalNumProducts() == 0)
            {
                System.out.println("\n== No products found in Category id = " + excludeCat + " ==");
            } else
            {
                System.out.println("\n== Error: products found in Category id = " + excludeCat
                        + " ==");
            }

        } catch (Exception e)
        {
            e.printStackTrace();
        }

    }

}
