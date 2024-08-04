def button_block(x,y, button_type = "top"):

    buttons_old = {
            0:((x - 150, 0 + 50), (x - 50, 0 + 150)), # button 0
            1:((0 + 40, y - 150), (0 + 140, y - 50)), # button 1
            2:((0 + 40, 0 + 50), (0 + 140, 0 + 150)), # button 2
            3:((x - 150, y - 150), (x - 50, y - 50)), # button 3
            4:((0, 0), (0 + 240, y)), # button 4
            5:((x - 240, 0), (x, y)), # button 5
        }

    buttons_full_body = {
            ((0 + 100, 0 + 100), (0 + 200, 0 + 200)), # button 0
            ((x - 200, 0 + 100), (x - 100, 0 + 200)),     # button 1
            ((0 + 150, y - 150), (0 + 250, y - 50)), # button 2
            ((x - 250, y - 150), (x - 150, y - 50)), # button 3
            ((0, 0), (0 + 260, y)), # button 4
            ((x - 260, 0), (x, y)), # button 5
        }

    buttons_big_corner = {
            0:((0 + 0, 0 + 0), (0 + 150, 0 + 150)), # button 0
            1:((x - 150, 0 ), (x, 0 + 150)), # button 1
            2:((0, y - 140), (0 + 150, y)), # button 2
            3:((x - 150, y - 140), (x - 0, y - 0)), # button 3
            4:((0, 0), (0 + 240, y)), # button 4
            5:((x - 240, 0), (x, y)), # button 5
        } 
    
    buttons_top= {
            ((0 + 0, 0 + 0), (0 + 120, 0 + 340)), # button 0
            ((x - 120, 0 ), (x, 0 + 340)), # button 1
            ((0 + 120, 0), (0 + 240, 0+340)), # button 2
            ((x - 240, 0), (x - 120, 0 + 340)), # button 3
            ((0, 0), (0 + 240, y)), # button 4
            ((x - 240, 0), (x, y)), # button 5
        } 
     
    
    if button_type == "old":
        return buttons_old
    elif button_type == "full":
        return buttons_full_body
    elif button_type == "corner":
        return buttons_big_corner
    else:
        return buttons_top
    