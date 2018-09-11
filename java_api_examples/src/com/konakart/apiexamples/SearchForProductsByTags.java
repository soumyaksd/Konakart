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
import com.konakart.app.Tag;
import com.konakart.app.TagGroup;
import com.konakart.appif.ProductIf;
import com.konakart.appif.ProductsIf;
import com.konakart.appif.TagGroupIf;
import com.konakart.appif.TagIf;

/**
 * This class shows how to call the KonaKart API to search for products using product tags. It
 * assumes that the KonaKart demo database is loaded.
 * 
 * Before running you may have to edit BaseApiExample.java to change the username and password used
 * to log into the engine. The default values are doe@konakart.com / password .
 */
public class SearchForProductsByTags extends BaseApiExample
{
    private static final String usage = "Usage: SearchForProductsByTags\n" + COMMON_USAGE;

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

            /*
             * In the demo database there are two tag groups. Id 1, MPAA Movie Rating and Id 2, DVD
             * Format. The MPAA Movie rating tag group has 5 tags which are G (id 1), PG (Id 2),
             * PG-13 (Id 3), R (Id 4) and NC-17 (Id 5). The DVD format has 2 tags which are Blu-Ray
             * (Id 6) and HB-DVD (Id 7)
             */

            /*
             * Search for all products with movie rating G or PG and in Blu-Ray format
             */
            TagIf tag1 = new Tag();
            tag1.setId(1);
            TagIf tag2 = new Tag();
            tag2.setId(2);
            TagIf tag6 = new Tag();
            tag6.setId(6);

            TagGroupIf tgMovieRating = new TagGroup();
            tgMovieRating.setId(1);
            tgMovieRating.setTags(new TagIf[]
            { tag1, tag2 });

            TagGroupIf tgDVDFormat = new TagGroup();
            tgDVDFormat.setId(2);
            tgDVDFormat.setTags(new TagIf[]
            { tag6 });

            // Create a product search object
            ProductSearch ps = new ProductSearch();
            ps.setTagGroups(new TagGroupIf[]
            { tgMovieRating, tgDVDFormat });

            // Create a data descriptor
            DataDescriptor datadesc = new DataDescriptor();
            datadesc.setOffset(0);
            datadesc.setLimit(DataDescConstants.MAX_ROWS);

            // Now we can search for the products.
            ProductsIf prods = eng.searchForProducts(/* sessionId */null, datadesc, ps,
                    DEFAULT_LANGUAGE);

            System.out.println("Total number of Products: " + prods.getTotalNumProducts());

            if (prods.getProductArray() != null)
            {
                for (int i = 0; i < prods.getProductArray().length; i++)
                {
                    ProductIf prod = prods.getProductArray()[i];
                    System.out.println("Product Name: " + prod.getName());
                }
            }

            // 4 products should be found and the output should look like:
            // Total number of Products: 4
            // Product Name: The Replacement Killers
            // Product Name: The Matrix
            // Product Name: Under Siege
            // Product Name: Under Siege 2 - Dark Territory

        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
